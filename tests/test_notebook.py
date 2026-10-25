"""Tests for notebooks/train_colab.ipynb and notebooks/cyv_colab.py (no GPU, no Colab needed)."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import struct
import sys
import wave
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NOTEBOOK = NB_DIR / "train_colab.ipynb"
SCHEMA = ROOT / "src" / "cyv" / "packs" / "voice_schema.json"
PIN = ROOT / "src" / "cyvd" / "engine_applio" / "pin.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("cyv_colab", NB_DIR / "cyv_colab.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cyv_colab"] = mod
    spec.loader.exec_module(mod)
    return mod


cyv_colab = _load_helper()


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code(nb):
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


# ------------------------------------------------------------------ notebook structure


def test_notebook_structure(nb):
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    assert isinstance(nb["metadata"], dict)
    assert nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        assert isinstance(c["metadata"], dict)
        assert isinstance(c["source"], (str, list))
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None  # committed clean


def test_code_cells_are_valid_python(nb):
    for src in _code(nb):
        ast.parse(src)  # no magics / shell escapes, so plain ast works


@pytest.mark.parametrize("bad", ["gradio", "share=True", "ngrok", "localtunnel"])
def test_no_ui_or_tunnels_in_code(nb, bad):
    for src in _code(nb):
        assert bad not in src.lower()


def test_helper_has_no_ui_or_tunnels():
    text = (NB_DIR / "cyv_colab.py").read_text().lower()
    for bad in ("gradio", "share=true", "ngrok", "localtunnel"):
        assert bad not in text


def test_commit_matches_pin(nb):
    pin = re.search(r'APPLIO_COMMIT\s*=\s*"([0-9a-f]{40})"', PIN.read_text()).group(1)
    found = set()
    for src in _code(nb):
        found.update(re.findall(r'APPLIO_COMMIT\s*=\s*\(?\s*"([0-9a-f]{40})"', src))
    assert found == {pin}


def test_notebook_uses_only_expected_applio_commands(nb):
    allowed = {"preprocess", "extract", "train", "index", "infer", "prerequisites"}
    text = "\n".join(_code(nb))
    used = set(re.findall(r'"core\.py",\s*"([a-z_-]+)"', text))
    assert used == allowed


def test_constraints_file_has_pins():
    lines = [
        ln.strip()
        for ln in (NB_DIR / "constraints.txt").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines and all(re.match(r"^[A-Za-z0-9_.-]+(==|<|>=)[^\s]+$", ln) for ln in lines)


# ------------------------------------------------------------------ voice.json


def _voice(**kw):
    args = dict(
        name="my_voice-1",
        engine_version="c817f1f83f689ef72e0dfeb1c17ae990b9248506",
        sample_rate=40000,
        base_voice="am_adam",
        pth_sha256="a" * 64,
        index_sha256="b" * 64,
        epochs=200,
        dataset_seconds=412.5,
    )
    args.update(kw)
    return cyv_colab.build_voice_json(**args)


def test_voice_json_matches_schema():
    voice = _voice()
    assert voice["tts_base"] == {"voice": "am_adam", "lang": "a"}
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator(schema).validate(voice)
    jsonschema.Draft202012Validator(schema).validate(_voice(epochs=None, dataset_seconds=None))


def test_voice_json_also_passes_repo_validator():
    try:
        from cyv.packs.manifest import validate_manifest
    except ImportError:
        pytest.skip("cyv not importable")
    validate_manifest(_voice())


@pytest.mark.parametrize("voice,lang", [("bf_emma", "b"), ("jf_alpha", "j"), ("zm_yunxi", "z")])
def test_lang_from_voice(voice, lang):
    assert cyv_colab.lang_from_voice(voice) == lang


@pytest.mark.parametrize("bad", ["", "Adam", "../x", "a" * 33, "has space"])
def test_bad_names_rejected(bad):
    with pytest.raises(cyv_colab.PackError):
        _voice(name=bad)


def test_bad_base_voice_rejected():
    with pytest.raises(cyv_colab.PackError):
        _voice(base_voice="adam")


# ------------------------------------------------------------------ dataset zip


def _wav_bytes(seconds=1.0, sr=16000, value=1000):
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<h", value) * int(sr * seconds))
    return buf.getvalue()


CONSENT = {
    "attestation": "This is my own voice.",
    "attestation_version": 1,
    "owner_name": "Test Person",
    "created_at": "2026-01-01T00:00:00Z",
}


def _make_dataset(tmp_path, *, n=8, seconds=10.0, consent=True, meta=None, extra=None):
    p = tmp_path / "dataset.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for i in range(n):
            zf.writestr(f"wavs/clip_{i:03d}.wav", _wav_bytes(seconds))
        if consent:
            zf.writestr("consent.json", json.dumps(CONSENT))
        zf.writestr("dataset_meta.json", json.dumps(meta or {"base_voice": "am_adam"}))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return p


def test_validate_dataset_ok(tmp_path):
    info = cyv_colab.validate_dataset_zip(_make_dataset(tmp_path))
    assert info["total_seconds"] == pytest.approx(80.0, abs=0.1)
    assert len(info["wav_names"]) == 8
    assert info["meta"]["base_voice"] == "am_adam"


def test_validate_dataset_missing_consent(tmp_path):
    with pytest.raises(cyv_colab.DatasetError, match="consent.json"):
        cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, consent=False))


def test_validate_dataset_too_short(tmp_path):
    with pytest.raises(cyv_colab.DatasetError, match="Only"):
        cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, n=2, seconds=5))


def test_validate_dataset_short_warns(tmp_path):
    info = cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, n=5, seconds=15))
    assert info["warnings"]


def test_validate_dataset_rejects_unexpected_and_traversal(tmp_path):
    with pytest.raises(cyv_colab.DatasetError):
        cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, extra={"notes.txt": "x"}))
    with pytest.raises(cyv_colab.DatasetError):
        cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, extra={"wavs/../../evil.wav": b"x"}))


def test_validate_dataset_bad_meta(tmp_path):
    with pytest.raises(cyv_colab.DatasetError, match="base_voice"):
        cyv_colab.validate_dataset_zip(_make_dataset(tmp_path, meta={"base_voice": "nope"}))


def test_validate_dataset_not_a_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(cyv_colab.DatasetError):
        cyv_colab.validate_dataset_zip(p)


def test_extract_dataset_flattens(tmp_path):
    z = _make_dataset(tmp_path)
    cyv_colab.extract_dataset(z, tmp_path / "ds", tmp_path / "meta")
    assert len(list((tmp_path / "ds").iterdir())) == 8
    assert all(p.suffix == ".wav" for p in (tmp_path / "ds").iterdir())
    assert (tmp_path / "meta" / "consent.json").is_file()


def test_pick_reference_clip(tmp_path):
    z = _make_dataset(tmp_path, seconds=6)
    assert cyv_colab.pick_reference_clip(z) == "wavs/clip_000.wav"


# ------------------------------------------------------------------ pack


def test_write_pack_exactly_four_members(tmp_path):
    pth, idx = tmp_path / "m.pth", tmp_path / "m.index"
    pth.write_bytes(b"weights" * 10)
    idx.write_bytes(b"index" * 10)
    voice = _voice(pth_sha256=cyv_colab.sha256_file(pth), index_sha256=cyv_colab.sha256_file(idx))
    consent_bytes = json.dumps(CONSENT, indent=1).encode()
    out = cyv_colab.write_pack(
        tmp_path / "my_voice-1.cyvpack.zip",
        pth_path=pth,
        index_path=idx,
        voice=voice,
        consent_bytes=consent_bytes,
    )
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["consent.json", "model.index", "model.pth", "voice.json"]
        assert zf.read("consent.json") == consent_bytes  # verbatim
        assert zf.read("model.pth") == pth.read_bytes()
        assert json.loads(zf.read("voice.json")) == voice
    assert not list(tmp_path.glob("*.part"))


def test_write_pack_refuses_hash_mismatch(tmp_path):
    pth, idx = tmp_path / "m.pth", tmp_path / "m.index"
    pth.write_bytes(b"a")
    idx.write_bytes(b"b")
    with pytest.raises(cyv_colab.PackError):
        cyv_colab.write_pack(
            tmp_path / "x.zip", pth_path=pth, index_path=idx, voice=_voice(), consent_bytes=b"{}"
        )


# ------------------------------------------------------------------ Applio outputs, resume


def test_find_final_weights_and_index(tmp_path):
    for n in (
        "v_10e_100s.pth",
        "v_200e_2000s.pth",
        "v_20e_200s.pth",
        "G_2333333.pth",
        "D_2333333.pth",
        "other_300e_1s.pth",
    ):
        (tmp_path / n).write_bytes(b"x")
    p, ep = cyv_colab.find_final_weights(tmp_path, "v")
    assert p.name == "v_200e_2000s.pth" and ep == 200
    assert cyv_colab.find_final_weights(tmp_path / "none", "v") == (None, 0)
    assert cyv_colab.find_index(tmp_path, "v") is None
    (tmp_path / "v.index").write_bytes(b"i")
    assert cyv_colab.find_index(tmp_path, "v").name == "v.index"


def test_sync_prepare_restore_roundtrip(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "sliced_audios").mkdir(parents=True)
    (local / "sliced_audios" / "0_0_0.wav").write_bytes(b"w")
    (local / "filelist.txt").write_text("f")
    for n in (
        "G_2333333.pth",
        "D_2333333.pth",
        "v_10e_100s.pth",
        "v_20e_200s.pth",
        "v_30e_300s.pth",
    ):
        (local / n).write_bytes(b"x")
    cyv_colab.save_prepared(local, drive)
    assert cyv_colab.sync_dir(local, drive) >= 4
    assert len(list(drive.glob("*e_*s.pth"))) == 2  # only newest weights kept
    assert cyv_colab.sync_dir(local, drive) == 0  # nothing changed

    fresh = tmp_path / "fresh"
    state = cyv_colab.restore_from_drive(drive, fresh)
    assert state == {"prepared": True, "checkpoint": True}
    assert (fresh / "filelist.txt").is_file() and (fresh / "sliced_audios" / "0_0_0.wav").is_file()
    assert (fresh / "G_2333333.pth").is_file()
    assert cyv_colab.restore_from_drive(tmp_path / "empty", tmp_path / "f2") == {
        "prepared": False,
        "checkpoint": False,
    }


def test_latest_loss_lines():
    log = "noise\nv | epoch=1 | step=5 | time=1\nv | epoch=2 | step=10 | time=2\nTraining has been successfully completed with 2 epoch\n"
    out = cyv_colab.latest_loss_lines(log, n=1)
    assert out[0].startswith("v | epoch=2") and out[1].startswith("Training has")


# ------------------------------------------------------------------ self-test heuristics


def _write_float_wav(path, x, sr=40000):
    import numpy as np

    data = np.asarray(x, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def test_analyze_wav_pass_and_fail(tmp_path):
    np = pytest.importorskip("numpy")
    t = np.arange(40000 * 2) / 40000
    good = tmp_path / "good.wav"
    _write_float_wav(good, 0.3 * np.sin(2 * np.pi * 220 * t))
    r = cyv_colab.analyze_wav(good, expected_seconds=2.2)
    assert r["passed"], r

    silent = tmp_path / "silent.wav"
    _write_float_wav(silent, np.zeros(80000))
    assert not cyv_colab.analyze_wav(silent)["passed"]

    nan = tmp_path / "nan.wav"
    _write_float_wav(nan, np.full(80000, np.nan))
    assert not cyv_colab.analyze_wav(nan)["passed"]

    assert not cyv_colab.analyze_wav(good, expected_seconds=20)["passed"]  # duration off
    assert not cyv_colab.analyze_wav(tmp_path / "missing.wav")["passed"]


def test_analyze_pcm16(tmp_path):
    pytest.importorskip("numpy")
    p = tmp_path / "pcm.wav"
    p.write_bytes(_wav_bytes(1.5, sr=16000, value=8000))
    assert cyv_colab.analyze_wav(p)["checks"]["readable"][0]


# ------------------------------------------------------------------ pinned helper hashes


def test_embedded_helper_hashes_match_files(nb):
    import hashlib

    text = "\n".join(_code(nb))
    block = re.search(r"HELPER_SHA256\s*=\s*\{(.*?)\}", text, re.S).group(1)
    embedded = dict(re.findall(r'"([\w.]+)":\s*"([0-9a-f]{64})"', block))
    assert set(embedded) == {"cyv_colab.py", "constraints.txt"}
    for name, digest in embedded.items():
        actual = hashlib.sha256((NB_DIR / name).read_bytes()).hexdigest()
        assert digest == actual, f"update HELPER_SHA256[{name!r}] in train_colab.ipynb to {actual}"


def test_install_cell_fails_closed_on_hash_mismatch(nb):
    text = "\n".join(_code(nb))
    assert "hashlib.sha256(blob)" in text and "Refusing to run" in text


def test_cleanup_cell_removes_selftest_and_features(nb):
    cell = next(s for s in _code(nb) if "YES_DELETE_MY_DATASET" in s)
    for needle in ("selftest_output.wav", "selftest_input.wav", '"extracted"', '"f0"', "_meta"):
        assert needle in cell
