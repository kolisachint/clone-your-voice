"""Helpers for voice_lab_colab.ipynb: cut a recorded batch into clips, check them, score them.

Standalone on purpose (numpy only), so Colab can load it from Drive without installing cyv.
The batch rules mirror src/cyv/dataset/batch.py; tests/test_lab.py checks that they agree.

Nothing here prints or logs transcript text. Reports hold file names and numbers only.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER = ("file", "lang", "style", "text")
NAME_RE = re.compile(r"^(?P<lang>[a-z]{2})_(?P<style>[a-z]{2,12})_(?P<num>[0-9]{3})$")
TTS_FOR_LANG = {"en": "f5", "hi": "indicf5", "mr": "indicf5"}
RVC_ONLY_STYLES = frozenset({"sing"})
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".aif", ".aiff", ".wma"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Clip lengths that suit F5-style training and reference cloning.
MIN_CLIP_S, MAX_CLIP_S = 1.0, 20.0
REF_MIN_S, REF_MAX_S = 4.0, 12.0
QA_MAX_CER = 0.35  # a clip whose transcript differs more than this is flagged for a listen

# Applio settings for B9 (docs/benchmarks/B9-voice-lab-batch-001.md). The index rate is the
# daemon's (src/cyvd/timbre/applio/engine.py), not Applio's own default of 0.3. Each variant is
# one conversion: "rvc" uses the daemon's protect of 0.5 (protection off), "rvc33" uses 0.33.
INDEX_RATE = 0.75
RVC_VARIANTS = {"rvc": 0.5, "rvc33": 0.33}  # name -> protect
NEUTRAL_STYLES = frozenset({"conv", "read"})


class LabError(Exception):
    pass


@dataclass(frozen=True)
class Line:
    file: str
    lang: str
    style: str
    text: str

    @property
    def engine(self) -> str | None:
        return None if self.style in RVC_ONLY_STYLES else TTS_FOR_LANG[self.lang]

    @property
    def take(self) -> str:
        return "sing" if self.style in RVC_ONLY_STYLES else self.lang


def parse_batch(content: str) -> list[Line]:
    rows = list(csv.reader(io.StringIO(content), delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows or tuple(c.strip() for c in rows[0]) != HEADER:
        raise LabError("batch file: the first line must be the header file/lang/style/text")
    out: list[Line] = []
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != 4:
            raise LabError(f"batch file line {n}: expected 4 tab-separated columns")
        x = Line(*(c.strip() for c in row))
        m = NAME_RE.fullmatch(x.file)
        if not m or x.lang not in TTS_FOR_LANG or (m["lang"], m["style"]) != (x.lang, x.style):
            raise LabError(f"batch file line {n}: bad file name or language {x.file!r}")
        if not x.text:
            raise LabError(f"batch file line {n}: empty text")
        if (x.lang == "en") == bool(_DEVANAGARI.search(x.text)):
            raise LabError(f"batch file line {n}: text is in the wrong script for {x.lang!r}")
        out.append(x)
    if not out or len({x.file for x in out}) != len(out):
        raise LabError("batch file: no lines, or a duplicate file name")
    return out


def takes(lines: list[Line]) -> dict[str, list[Line]]:
    out: dict[str, list[Line]] = {}
    for x in lines:
        out.setdefault(x.take, []).append(x)
    return out


def take_name(batch: str, take: str) -> str:
    return f"batch-{batch}_{take}"


# ------------------------------------------------------------------ finding the uploads


@dataclass
class Uploads:
    takes: dict[str, Path] = field(default_factory=dict)  # take -> long recording
    clips: dict[str, Path] = field(default_factory=dict)  # line file -> its own recording
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # line files with no audio at all


def find_uploads(folder: Path, batch: str, lines: list[Line]) -> Uploads:
    by_stem: dict[str, list[Path]] = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in AUDIO_EXTS:
            by_stem.setdefault(p.stem, []).append(p)
    up = Uploads()
    groups = takes(lines)
    known = {take_name(batch, t) for t in groups} | {x.file for x in lines}
    for stem, paths in by_stem.items():
        if len(paths) > 1:
            up.problems.append(f"{stem}: more than one file with this name, keep one")
        elif stem not in known:
            up.problems.append(f"{paths[0].name}: not part of batch {batch}, ignored")
    for take, group in groups.items():
        name = take_name(batch, take)
        if len(by_stem.get(name, [])) == 1:
            up.takes[take] = by_stem[name][0]
        for x in group:
            if len(by_stem.get(x.file, [])) == 1:
                up.clips[x.file] = by_stem[x.file][0]  # a single-line file wins over a take
            elif take not in up.takes:
                up.missing.append(x.file)
    return up


# ------------------------------------------------------------------ cutting a long take


def speech_regions(
    samples: np.ndarray,
    sr: int,
    min_silence: float = 0.5,
    min_speech: float = 0.15,
    floor_db: float = -45.0,
    below_peak_db: float = 35.0,
) -> list[tuple[float, float]]:
    """Stretches of speech separated by at least `min_silence` seconds of quiet."""
    hop = int(sr * 0.02)
    n = len(samples) // hop
    if n == 0:
        return []
    frames = samples[: n * hop].astype(np.float64).reshape(n, hop)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    loud = db > max(floor_db, float(np.percentile(db, 98)) - below_peak_db)
    regions: list[tuple[int, int]] = []
    start = None
    quiet = 0
    gap = int(min_silence / 0.02)
    for i, on in enumerate(loud):
        if on:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap:
                regions.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        regions.append((start, n - quiet))
    return [(a * 0.02, b * 0.02) for a, b in regions if (b - a) * 0.02 >= min_speech]


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}


def parse_number(text: str) -> int | None:
    """'Seven.' -> 7, '21' -> 21, 'twenty-one' -> 21. Anything else -> None."""
    t = re.sub(r"[^a-z0-9 -]", "", text.lower()).strip()
    t = re.sub(r"^(number|no)\s+", "", t)
    if t.isdigit():
        return int(t)
    parts = [p for p in re.split(r"[\s-]+", t) if p]
    if len(parts) == 1 and parts[0] in _ONES:
        return _ONES.index(parts[0])
    if parts and parts[0] in _TENS:
        if len(parts) == 1:
            return _TENS[parts[0]]
        if len(parts) == 2 and parts[1] in _ONES[1:10]:
            return _TENS[parts[0]] + _ONES.index(parts[1])
    return None


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds, relative to the audio given to the transcriber
    end: float


# (samples at 16 kHz, language) -> words with timestamps
Transcriber = Callable[[np.ndarray, str], list[Word]]

# Give this to the transcriber as its prompt when it listens for numbers. Without it, Whisper
# often hears a lone spoken number as some other word (tested with faster-whisper small).
NUMBER_PROMPT = " ".join(w.capitalize() + "." for w in _ONES[1:] + list(_TENS))
MARKER_LISTEN_S = 2.0  # a spoken number sits at the very start of its region
MIN_LINE_SPEECH_S = 1.0  # a new number counts only after this much of the current line


@dataclass
class Cut:
    spans: dict[int, tuple[float, float]] = field(default_factory=dict)  # line number -> seconds
    retakes: dict[int, int] = field(default_factory=dict)  # line number -> extra takes dropped
    problems: list[str] = field(default_factory=list)


def _marker(words: list[Word], expected: set[int]) -> tuple[int, float] | None:
    """A spoken line number at the start of a region: (number, where the line starts)."""
    if not words:
        return None
    for k in (2, 1):  # "twenty one" is two words
        if len(words) >= k:
            n = parse_number(" ".join(w.text for w in words[:k]))
            if n in expected:
                return n, words[k - 1].end
    return None


def cut_take(
    samples16k: np.ndarray,
    count: int,
    transcribe: Transcriber,
    pad_before: float = 0.12,
    pad_after: float = 0.3,  # soft word endings fall below the speech threshold
) -> Cut:
    """Find lines 1..count in one long recording, using the spoken numbers.

    A number starts a take of that line. Saying the number again starts a new take, and the
    last take wins. The spoken number itself is not part of the clip.
    """
    sr = 16000
    regions = speech_regions(samples16k, sr)
    cut = Cut()
    current: int | None = None
    spoken = 0.0  # seconds of line speech since the last number
    takes_seen: dict[int, int] = {}
    for a, b in regions:
        m = None
        # Two guards keep a word inside a line from being read as a number: only the next few
        # numbers (or a retake of the current one) count, and only once the current line has
        # been spoken for a while. The first word of a line never ends it.
        if current is None or spoken >= MIN_LINE_SPEECH_S:
            nxt = (current or 0) + 1
            expected = {n for n in range(nxt, nxt + 3) if n <= count}
            if current is not None:
                expected.add(current)
            seg = samples16k[int(a * sr) : int(min(b, a + MARKER_LISTEN_S) * sr)]
            m = _marker(transcribe(seg, "en"), expected)
        if m is not None:
            current, after = m
            takes_seen[current] = takes_seen.get(current, 0) + 1
            line_start = a + after
            if b - line_start < 0.4:  # the number was alone in its region
                cut.spans.pop(current, None)
                spoken = 0.0
                continue
            cut.spans[current] = (line_start, b)
            spoken = b - line_start
            continue
        if current is None:
            cut.problems.append(f"speech at {a:.1f} s before the first number, ignored")
            continue
        s, _ = cut.spans.get(current, (a, b))
        cut.spans[current] = (s, b)
        spoken += b - a
    for n, (s, e) in list(cut.spans.items()):
        cut.spans[n] = (max(0.0, s - pad_before), min(len(samples16k) / sr, e + pad_after))
    cut.retakes = {n: k - 1 for n, k in takes_seen.items() if k > 1}
    for n in range(1, count + 1):
        if n not in cut.spans:
            cut.problems.append(f"line {n}: not found (was its number said in English?)")
        else:
            s, e = cut.spans[n]
            if not MIN_CLIP_S <= e - s <= MAX_CLIP_S:
                cut.problems.append(f"line {n}: {e - s:.1f} s long, check it by ear")
    return cut


# ------------------------------------------------------------------ checks and scores


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after normalizing case, punctuation and spaces."""
    r, h = normalize_text(reference), normalize_text(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def _usable_ref(x: Line, qa: dict[str, dict], seconds: dict[str, float]) -> bool:
    return (
        x.engine is not None
        and qa.get(x.file, {}).get("ok", False)
        and REF_MIN_S <= seconds.get(x.file, 0.0) <= REF_MAX_S
    )


def pick_reference(
    target: Line, lines: list[Line], qa: dict[str, dict], seconds: dict[str, float]
) -> Line | None:
    """Another clip of the same language to clone from: same style first, then lowest CER."""
    pool = [
        x
        for x in lines
        if x.file != target.file and x.lang == target.lang and _usable_ref(x, qa, seconds)
    ]
    if not pool:
        return None
    return min(pool, key=lambda x: (x.style != target.style, qa[x.file]["cer"], x.file))


def emotion_set(
    lines: list[Line], qa: dict[str, dict], seconds: dict[str, float]
) -> dict[str, tuple[Line, list[Line]]]:
    """B9b: per language, one neutral line and the clips to speak it from.

    The clips are one calm reference (conv or read) and every usable clip in an emotion style.
    The same words spoken from each shows whether the speech model copies the reference's tone.
    A language without a calm reference, an emotion clip and a second neutral line is left out.
    """
    out: dict[str, tuple[Line, list[Line]]] = {}
    for lang in sorted({x.lang for x in lines}):
        mine = [x for x in lines if x.lang == lang]
        calm = [x for x in mine if x.style in NEUTRAL_STYLES and _usable_ref(x, qa, seconds)]
        moods = [x for x in mine if x.style not in NEUTRAL_STYLES and _usable_ref(x, qa, seconds)]
        if not calm or not moods:
            continue
        ref = min(calm, key=lambda x: (qa[x.file]["cer"], x.file))
        targets = [x for x in mine if x.style in NEUTRAL_STYLES and x.file != ref.file]
        if targets:
            out[lang] = (targets[0], [ref, *sorted(moods, key=lambda x: (x.style, x.file))])
    return out


def listening_sheet(title: str, question: str, groups: dict[str, list[str]]) -> str:
    """A form to fill in while listening: one row per group, file names only."""
    out = [f"# {title}", "", question, "", "| group | files | your answer |", "|---|---|---|"]
    out += [f"| {g} | {', '.join(files)} | |" for g, files in groups.items()]
    return "\n".join(out) + "\n"


def report_markdown(batch: str, rows: list[dict]) -> str:
    """Results table: file names and numbers only, never the text."""

    def f(v: object) -> str:
        return f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))

    scored = ["tts", *RVC_VARIANTS]
    cols = ["file", "engine", "ref", "qa_cer"] + [
        f"{v}_{m}" for v in scored for m in ("cer", "sim")
    ]
    out = [
        f"# Batch {batch}: results",
        "",
        "- `qa_cer`: how far your recording is from its line (speech recognition). Lower is better.",
        "- `*_cer`: the same for the cloned speech. `tts` is before your voice model; "
        + ", ".join(f"`{v}` is after it with protect {p}" for v, p in RVC_VARIANTS.items())
        + f", index rate {INDEX_RATE}.",
        "- `*_sim`: how much the clone sounds like your real recording of the line "
        "(speaker similarity, 0 to 1). Higher is better.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    out += ["| " + " | ".join(f(r.get(c)) for c in cols) + " |" for r in rows]
    for engine in sorted({r["engine"] for r in rows if r.get("engine")}):
        sub = [r for r in rows if r.get("engine") == engine]
        for key in [f"{v}_{m}" for m in ("sim", "cer") for v in scored]:
            vals = [r[key] for r in sub if isinstance(r.get(key), float)]
            if vals:
                out.append(
                    f"- {engine} mean {key}: {sum(vals) / len(vals):.2f} ({len(vals)} lines)"
                )
    return "\n".join(out) + "\n"
