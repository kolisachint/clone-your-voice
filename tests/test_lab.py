"""Tests for notebooks/cyv_lab.py (no GPU, no models: the transcriber is faked)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cyv.dataset import batch as cyv_batch

ROOT = Path(__file__).resolve().parents[1]
BATCHES = sorted((ROOT / "docs" / "guides" / "batches").glob("batch-*.tsv"))


def _load():
    spec = importlib.util.spec_from_file_location("cyv_lab", ROOT / "notebooks" / "cyv_lab.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cyv_lab"] = mod
    spec.loader.exec_module(mod)
    return mod


lab = _load()
SR = 16000


def test_batch_rules_match_cyv() -> None:
    assert lab.TTS_FOR_LANG == __import__("cyv.languages").languages.TTS_FOR_LANG
    assert lab.RVC_ONLY_STYLES == __import__("cyv.languages").languages.RVC_ONLY_STYLES
    assert lab.HEADER == cyv_batch.HEADER and lab.NAME_RE.pattern == cyv_batch.NAME_RE.pattern
    assert BATCHES
    for b in BATCHES:
        ours = lab.parse_batch(b.read_text(encoding="utf-8"))
        theirs = cyv_batch.read_manifest(b)
        assert [(x.file, x.lang, x.style, x.text, x.engine) for x in ours] == [
            (x.file, x.lang, x.style, x.text, x.engine) for x in theirs
        ]
        assert {k: [x.file for x in v] for k, v in lab.takes(ours).items()} == {
            k: [x.file for x in v] for k, v in cyv_batch.takes(theirs).items()
        }


@pytest.mark.parametrize(
    "bad",
    [
        "file\tlang\tstyle\ttext\nen_conv_001\ten\tconv\tनमस्ते\n",
        "file\tlang\tstyle\ttext\nhi_conv_001\thi\tconv\tnamaste\n",
        "file\tlang\tstyle\ttext\nen_conv_001\ten\tread\tHello.\n",
        "nope\n",
    ],
)
def test_parse_batch_rejects(bad: str) -> None:
    with pytest.raises(lab.LabError):
        lab.parse_batch(bad)


@pytest.mark.parametrize(
    ("text", "n"),
    [("Seven.", 7), ("21", 21), ("twenty-one", 21), ("Twenty one", 21), ("Number 3", 3),
     ("thirty", 30), ("the", None), ("No, I have", None), ("", None)],
)  # fmt: skip
def test_parse_number(text: str, n: int | None) -> None:
    assert lab.parse_number(text) == n


def _tone(seconds: float, freq: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


def test_speech_regions() -> None:
    x = np.concatenate([_gap(0.5), _tone(1.0), _gap(1.0), _tone(2.0), _gap(0.2), _tone(1.0)])
    r = lab.speech_regions(x, SR)
    assert len(r) == 2  # the 0.2 s pause does not split a line
    assert r[0][0] == pytest.approx(0.5, abs=0.05) and r[1][1] == pytest.approx(5.7, abs=0.05)


class FakeASR:
    """Answers each region by its loudness: every test region gets its own amplitude."""

    def __init__(self, answers: dict[float, list[tuple[str, float, float]]]) -> None:
        self.answers = answers
        self.heard: list[float] = []
        self.langs: list[str] = []

    def __call__(self, seg: np.ndarray, lang: str) -> list:
        assert len(seg) <= lab.MARKER_LISTEN_S * SR + 1
        amp = round(float(np.max(np.abs(seg))), 2)
        self.heard.append(amp)
        self.langs.append(lang)
        return [lab.Word(*w) for w in self.answers.get(amp, [])]


def _take(parts: list[tuple[float, float]]) -> np.ndarray:
    """(seconds, amplitude) pairs; amplitude 0 is silence."""
    return np.concatenate([_tone(s, amp=a) if a else _gap(s) for s, a in parts])


def test_cut_take_numbers_retakes_and_merged_marker() -> None:
    # "one" | line 1 | "two" | bad take | "two" | line 2 (with a pause inside) | "three line 3"
    audio = _take(
        [
            (0.5, 0), (0.5, 0.21), (1.0, 0), (3.0, 0.22), (1.5, 0),
            (0.5, 0.23), (1.0, 0), (2.0, 0.24), (1.0, 0),
            (0.5, 0.25), (1.0, 0), (2.0, 0.26), (0.8, 0), (1.5, 0.27), (1.5, 0),
            (4.0, 0.28), (0.5, 0),
        ]
    )  # fmt: skip
    asr = FakeASR(
        {
            0.21: [("One.", 0.0, 0.4)],
            0.23: [("Two.", 0.0, 0.4)],
            0.24: [("Wait", 0.0, 0.4)],
            0.25: [("two", 0.0, 0.4)],
            0.27: [("Okay", 0.0, 0.4)],
            0.28: [("Three.", 0.0, 0.5), ("We", 0.9, 1.1)],
        }
    )
    cut = lab.cut_take(audio, 3, asr)
    assert set(asr.langs) == {"en"}
    # The first region after a number is never asked: a line's first word cannot end it.
    assert asr.heard == [0.21, 0.23, 0.25, 0.27, 0.28]
    assert sorted(cut.spans) == [1, 2, 3]
    assert cut.retakes == {2: 1}
    s1, e1 = cut.spans[1]
    assert s1 == pytest.approx(2.0 - 0.12, abs=0.05) and e1 == pytest.approx(5.0 + 0.3, abs=0.06)
    s2, e2 = cut.spans[2]  # the retake, both halves, not the first take
    assert s2 == pytest.approx(12.5 - 0.12, abs=0.05) and e2 == pytest.approx(16.8 + 0.3, abs=0.06)
    s3, _ = cut.spans[3]  # the number is cut off the front
    assert s3 == pytest.approx(18.3 + 0.5 - 0.12, abs=0.05)
    assert cut.problems == []


def test_cut_take_ignores_numbers_said_too_soon() -> None:
    # "one" | a short first word of line 1, misheard as "two" | rest of line 1
    audio = _take([(0.5, 0), (0.5, 0.21), (1.0, 0), (0.4, 0.22), (0.6, 0), (2.0, 0.23)])
    asr = FakeASR({0.21: [("One.", 0.0, 0.4)], 0.22: [("Two.", 0.0, 0.3)]})
    cut = lab.cut_take(audio, 2, asr)
    assert 0.22 not in asr.heard
    assert sorted(cut.spans) == [1]
    assert cut.spans[1][0] == pytest.approx(2.0 - 0.12, abs=0.05)


def test_cut_take_reports_missing_and_ignores_far_numbers() -> None:
    audio = _take([(0.5, 0), (2.0, 0.21), (1.0, 0), (0.5, 0.22), (1.0, 0), (2.0, 0.23)])
    asr = FakeASR({0.21: [("hello", 0.0, 0.3)], 0.22: [("Nine", 0.0, 0.3)]})
    cut = lab.cut_take(audio, 2, asr)
    assert cut.spans == {}
    assert any("before the first number" in p for p in cut.problems)
    assert any("line 1: not found" in p for p in cut.problems)


def test_number_prompt_lists_words() -> None:
    assert lab.NUMBER_PROMPT.startswith("One. Two. Three.") and "Twenty." in lab.NUMBER_PROMPT


def test_cer() -> None:
    assert lab.cer("Hello, World!", "hello world") == 0.0
    assert lab.cer("abcd", "abed") == pytest.approx(0.25)
    assert lab.cer("नमस्ते।", "नमस्ते") == 0.0
    assert lab.cer("", "") == 0.0 and lab.cer("", "x") == 1.0


def _lines():
    return [
        lab.Line("en_conv_001", "en", "conv", "Hey there, how are you?"),
        lab.Line("en_conv_002", "en", "conv", "Wait, really?"),
        lab.Line("en_read_001", "en", "read", "The old library."),
        lab.Line("hi_conv_001", "hi", "conv", "अरे यार"),
        lab.Line("hi_sing_001", "hi", "sing", "सा रे ग म"),
    ]


def test_pick_reference() -> None:
    lines = _lines()
    qa = {x.file: {"ok": True, "cer": 0.1} for x in lines}
    secs = dict.fromkeys(qa, 6.0)
    t = lines[0]
    assert lab.pick_reference(t, lines, qa, secs).file == "en_conv_002"  # same style first
    qa["en_conv_002"]["ok"] = False
    assert lab.pick_reference(t, lines, qa, secs).file == "en_read_001"
    secs["en_read_001"] = 2.0  # too short to clone from
    assert lab.pick_reference(t, lines, qa, secs) is None
    assert lab.pick_reference(lines[3], lines, qa, secs) is None  # sing clips are never refs


def test_emotion_set() -> None:
    lines = [
        *_lines(),
        lab.Line("en_sad_001", "en", "sad", "I miss it."),
        lab.Line("en_angry_001", "en", "angry", "Stop it now!"),
        lab.Line("hi_happy_001", "hi", "happy", "वाह"),
    ]
    qa = {x.file: {"ok": True, "cer": 0.1} for x in lines}
    qa["en_read_001"]["cer"] = 0.05  # the cleanest calm clip is the calm reference
    secs = dict.fromkeys(qa, 6.0)
    emo = lab.emotion_set(lines, qa, secs)
    target, refs = emo["en"]
    assert target.file == "en_conv_001"
    assert [r.file for r in refs] == ["en_read_001", "en_angry_001", "en_sad_001"]
    assert "hi" not in emo  # one calm line only: nothing left to speak
    secs["en_angry_001"] = secs["en_sad_001"] = 2.0  # too short to clone from
    assert lab.emotion_set(lines, qa, secs) == {}


def test_listening_sheet_has_names_not_text() -> None:
    sheet = lab.listening_sheet("T", "Q?", {"en_conv_001": ["A", "B"]})
    assert "| en_conv_001 | A, B | |" in sheet


def test_rvc_settings_match_daemon() -> None:
    import re

    engine = (ROOT / "src" / "cyvd" / "timbre" / "applio" / "engine.py").read_text()
    assert f"INDEX_RATE = {lab.INDEX_RATE}" in engine
    assert re.search(rf"PROTECT = {lab.RVC_VARIANTS['rvc']}\b", engine)
    assert all(0 <= p <= 0.5 for p in lab.RVC_VARIANTS.values())  # Applio's allowed range
    assert '"--protect", protect' in _nb_code() and '"--index-rate", lab.INDEX_RATE' in _nb_code()


def test_find_uploads(tmp_path: Path) -> None:
    lines = _lines()
    for name in ("batch-001_en.m4a", "hi_conv_001.wav", "stray.wav", "batch-001_sing.wav"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "batch-001.tsv").write_text("not audio")
    up = lab.find_uploads(tmp_path, "001", lines)
    assert up.takes == {
        "en": tmp_path / "batch-001_en.m4a",
        "sing": tmp_path / "batch-001_sing.wav",
    }
    assert up.clips == {"hi_conv_001": tmp_path / "hi_conv_001.wav"}
    assert up.missing == []
    assert any("stray.wav" in p for p in up.problems)
    (tmp_path / "hi_conv_001.wav").unlink()
    assert lab.find_uploads(tmp_path, "001", lines).missing == ["hi_conv_001"]


def test_report_has_numbers_not_text() -> None:
    rows = [
        {"file": "en_conv_001", "engine": "f5", "ref": "en_conv_002", "qa_cer": 0.05,
         "tts_cer": 0.1, "tts_sim": 0.6, "rvc_cer": 0.12, "rvc_sim": 0.8,
         "rvc33_cer": 0.11, "rvc33_sim": 0.7},
        {"file": "hi_sing_001", "engine": None, "qa_cer": None},
    ]  # fmt: skip
    md = lab.report_markdown("001", rows)
    assert "| en_conv_001 | f5 | en_conv_002 | 0.05 |" in md
    assert "f5 mean rvc_sim: 0.80 (1 lines)" in md
    assert "f5 mean rvc33_sim: 0.70 (1 lines)" in md
    assert "| 0.12 | 0.80 | 0.11 | 0.70 |" in md
    assert "Hey" not in md


# ------------------------------------------------------------------ the notebook

NOTEBOOK = ROOT / "notebooks" / "voice_lab_colab.ipynb"


def _nb_code() -> str:
    import json

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None  # committed clean
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_pins_match_repo() -> None:
    import re

    src = _nb_code()
    pin = (ROOT / "src" / "cyvd" / "timbre" / "applio" / "pin.py").read_text()
    commit = re.search(r'APPLIO_COMMIT\s*=\s*"([0-9a-f]{40})"', pin).group(1)
    assert re.findall(r'APPLIO_COMMIT = "([0-9a-f]{40})"', src) == [commit]
    import hashlib

    sha = hashlib.sha256((ROOT / "notebooks" / "constraints.txt").read_bytes()).hexdigest()
    assert f'CONSTRAINTS_SHA256 = "{sha}"' in src
    # Every downloaded model is pinned to one revision.
    for name in ("ASR_REVISION", "INDIC_REVISION", "SV_REVISION"):
        assert re.search(rf'{name} = "[0-9a-f]{{40}}"', src), name
    assert 'F5_VERSION = "f5-tts==1.1.22"' in src and "faster-whisper==1.2.1" in src


def test_notebook_never_prints_text() -> None:
    src = _nb_code()
    for bad in ("print(x.text", "print(text", "print(ref.text", 'print(j["text"]'):
        assert bad not in src
    assert "lab.NUMBER_PROMPT" in src  # numbers are listened for with the prompt
