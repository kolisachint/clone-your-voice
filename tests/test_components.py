"""Daemon components: light unit tests plus opt-in end-to-end runs.

Heavy tests need the engine venv, base models and a voice pack, and run only with
``CYV_E2E=1`` plus:
  CYV_E2E_MODELS  models dir laid out per src/cyv/models.toml (default: paths.models_dir())
  CYV_E2E_VOICE   a voice pack dir (model.pth, model.index, voice.json)
  CYV_APPLIO_DIR  Applio checkout at the pinned commit
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

from cyv import paths
from cyvd.interfaces import Audio
from cyvd.watermark import CYV_MESSAGE, dsp, get_watermarker, pack, unpack

MODELS_TOML = Path(__file__).parents[1] / "src" / "cyv" / "models.toml"


def tone(freq: float, sr: int, seconds: float = 1.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- message layout ----------------------------------------------------------------


def test_message_layout() -> None:
    assert 0 <= CYV_MESSAGE < 1 << 16
    version, flags, crc_ok = unpack(CYV_MESSAGE)
    assert (version, flags, crc_ok) == (1, 0b0001, True)
    assert CYV_MESSAGE >> 12 == 1
    assert CYV_MESSAGE >> 8 == 0x11


def test_message_crc_detects_single_bit_flips() -> None:
    for bit in range(16):
        assert not unpack(CYV_MESSAGE ^ (1 << bit))[2]


def test_pack_rejects_wide_fields() -> None:
    with pytest.raises(ValueError):
        pack(version=16)
    with pytest.raises(ValueError):
        pack(flags=-1)


def test_bits_roundtrip() -> None:
    bits = dsp.int_to_bits(CYV_MESSAGE, 16)
    assert len(bits) == 16 and bits[:4] == [0, 0, 0, 1]
    assert dsp.bits_to_int(np.array(bits)) == CYV_MESSAGE


def test_unknown_watermarker() -> None:
    with pytest.raises(ValueError):
        get_watermarker("nope")


# -- resample / residual math ------------------------------------------------------


@pytest.mark.parametrize(("sr_in", "sr_out"), [(24000, 16000), (16000, 24000), (48000, 16000)])
def test_resample_length_and_tone(sr_in: int, sr_out: int) -> None:
    x = tone(1000, sr_in)
    y = dsp.resample(x, sr_in, sr_out)
    assert len(y) == round(len(x) * sr_out / sr_in)
    np.testing.assert_allclose(y, tone(1000, sr_out), atol=1e-3)


def test_add_residual_keeps_original_bandwidth() -> None:
    sr = 48000
    x = tone(12000, sr)  # above AudioSeal's 8 kHz Nyquist
    residual = 0.01 * tone(300, 16000)
    y = dsp.add_residual(x, sr, residual)
    assert y.shape == x.shape and y.dtype == np.float32
    spec = np.abs(np.fft.rfft(y - x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    assert freqs[spec.argmax()] == pytest.approx(300, abs=2)
    assert spec[freqs > 8000].max() < 1e-3 * spec.max()
    np.testing.assert_allclose(y - x, 0.01 * tone(300, sr), atol=1e-4)


def test_add_residual_pads_and_clips() -> None:
    x = np.full(1000, 0.99, dtype=np.float32)
    y = dsp.add_residual(x, 16000, np.full(900, 0.5, dtype=np.float32))
    assert len(y) == 1000 and y.max() <= 1.0


# -- AudioSeal wrapper with fake models --------------------------------------------


class FakeGen:
    def get_watermark(self, x, sample_rate, message):  # noqa: ANN001
        import torch

        n = x.shape[-1]
        t = torch.arange(n) / 16000
        bit_sum = message.float().sum()
        return (0.01 * torch.sin(2 * torch.pi * 440 * t) * (1 + bit_sum * 0))[None, None, :]


class FakeDet:
    normalizer = None

    def detector(self, x):  # noqa: ANN001
        import torch

        frames = x.shape[-1] // 320
        out = torch.zeros(1, 18, frames)
        out[:, 1] = 5.0  # class "watermarked"
        for i, b in enumerate(dsp.int_to_bits(CYV_MESSAGE, 16)):
            out[:, 2 + i] = 4.0 if b else -4.0
        return out


@pytest.mark.torch
def test_audioseal_wrapper_with_fake_models() -> None:
    pytest.importorskip("torch")
    from cyvd.watermark.audioseal_wm import AudioSealWatermarker

    wm = AudioSealWatermarker(models_dir=Path("/nonexistent"))
    wm._gen, wm._det = FakeGen(), FakeDet()
    sr = 24000
    audio = Audio(tone(1000, sr, 2.0), sr)
    marked = wm.embed(audio, CYV_MESSAGE)
    assert marked.sample_rate == sr and marked.samples.shape == audio.samples.shape
    diff = marked.samples - audio.samples
    assert 0.005 < np.abs(diff).max() < 0.02
    prob, msg = wm.detect(marked)
    assert prob == 1.0 and msg == CYV_MESSAGE


@pytest.mark.torch
def test_audioseal_missing_checkpoints_message(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("audioseal")
    from cyvd.watermark.audioseal_wm import AudioSealWatermarker

    with pytest.raises(FileNotFoundError, match="cyv setup"):
        AudioSealWatermarker(models_dir=tmp_path).embed(Audio(tone(440, 16000), 16000), CYV_MESSAGE)


# -- engine / tts guards -----------------------------------------------------------


def _voice_dir(tmp_path: Path, **overrides: object) -> Path:
    d = tmp_path / "v"
    d.mkdir()
    for name in ("model.pth", "model.index"):
        (d / name).write_bytes(b"x")
    meta = {
        "engine": "applio-rvc",
        "rvc_version": "v2",
        "embedder": "contentvec",
        "f0_method": "rmvpe",
    }
    (d / "voice.json").write_text(json.dumps({**meta, **overrides}))
    return d


def test_engine_rejects_bad_voice_dirs(tmp_path: Path) -> None:
    from cyvd.engine_applio import ApplioEngine, EngineError

    eng = ApplioEngine(tmp_path, device="cpu")
    assert eng.device == "cpu"
    with pytest.raises(EngineError, match="missing"):
        eng.load(tmp_path / "nope")
    with pytest.raises(EngineError, match="embedder"):
        eng.load(_voice_dir(tmp_path, embedder="spin"))


def test_engine_convert_requires_load(tmp_path: Path) -> None:
    from cyvd.engine_applio import ApplioEngine, EngineError

    with pytest.raises(EngineError, match="no voice"):
        ApplioEngine(tmp_path, device="cpu").convert(Audio(tone(200, 24000), 24000), 0)


def test_engine_reports_missing_applio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cyvd.engine_applio import ApplioEngine, EngineError

    monkeypatch.setenv("CYV_APPLIO_DIR", str(tmp_path / "none"))
    eng = ApplioEngine(tmp_path, device="cpu")
    with pytest.raises(EngineError, match="Applio checkout"):
        eng.load(_voice_dir(tmp_path))


def test_applio_dir_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cyvd.engine_applio.engine import applio_dir

    monkeypatch.delenv("CYV_APPLIO_DIR", raising=False)
    monkeypatch.setenv("CYV_HOME", str(tmp_path))
    assert applio_dir() == tmp_path / "applio"
    monkeypatch.setenv("CYV_APPLIO_DIR", str(tmp_path / "x"))
    assert applio_dir() == tmp_path / "x"


def test_index_cache_reads_and_reconstructs_once(tmp_path: Path) -> None:
    from cyvd.engine_applio.engine import _IndexCache

    class FakeIndex:
        ntotal = 3
        calls = 0

        def reconstruct_n(self, start: int, n: int) -> np.ndarray:
            FakeIndex.calls += 1
            return np.arange(6, dtype=np.float32).reshape(3, 2)[start : start + n]

    class FakeFaiss:
        reads = 0

        def read_index(self, path: str) -> FakeIndex:
            FakeFaiss.reads += 1
            return FakeIndex()

    f = tmp_path / "a.index"
    f.write_bytes(b"1")
    cache = _IndexCache(FakeFaiss())
    for _ in range(3):
        idx = cache.read_index(str(f))
        assert idx.reconstruct_n(0, idx.ntotal).shape == (3, 2)
    assert (FakeFaiss.reads, FakeIndex.calls) == (1, 1)


def test_kokoro_validates_voice(tmp_path: Path) -> None:
    from cyvd.tts_kokoro import KokoroTTS, TTSError

    tts = KokoroTTS(device="cpu", models_dir=tmp_path)
    for bad in ("../x", "af_heart.pt", "AF_HEART", ""):
        with pytest.raises(TTSError, match="invalid voice"):
            tts.synth("hi", bad, "a")
    with pytest.raises(TTSError, match="cyv setup"):
        tts.synth("hi", "af_heart", "a")


# -- models.toml -------------------------------------------------------------------


def test_models_toml_is_well_formed() -> None:
    entries = tomllib.loads(MODELS_TOML.read_text())["model"]
    assert entries
    dests = [e["dest"] for e in entries]
    assert len(set(dests)) == len(dests)
    for e in entries:
        assert set(e) >= {"name", "url", "sha256", "size_bytes", "dest", "used_by"}
        assert len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0
        assert e["size_bytes"] > 0
        assert e["url"].startswith("https://huggingface.co/") and "/resolve/" in e["url"]
        p = Path(e["dest"])
        assert not p.is_absolute() and ".." not in p.parts
    used = {e["used_by"] for e in entries}
    assert used == {"applio-rvc", "kokoro", "audioseal"}
    assert "applio/predictors/rmvpe.pt" in dests and "kokoro/kokoro-v1_0.pth" in dests


# -- slow end-to-end ---------------------------------------------------------------

e2e = pytest.mark.skipif(os.environ.get("CYV_E2E") != "1", reason="set CYV_E2E=1 for real models")


def _e2e_paths() -> tuple[Path, Path]:
    models = Path(os.environ.get("CYV_E2E_MODELS", paths.models_dir()))
    voice = os.environ.get("CYV_E2E_VOICE")
    if not voice:
        pytest.skip("CYV_E2E_VOICE not set")
    for dep in ("torch", "kokoro", "audioseal", "soxr", "faiss"):
        pytest.importorskip(dep)
    return models, Path(voice)


@pytest.mark.slow
@e2e
def test_e2e_pipeline(tmp_path: Path) -> None:
    models, voice = _e2e_paths()
    sf = pytest.importorskip("soundfile")
    from cyvd.engine_applio import ApplioEngine
    from cyvd.tts_kokoro import KokoroTTS

    guide = KokoroTTS(models_dir=models).synth("This is a short end to end test.", "af_heart", "a")
    assert guide.sample_rate == 24000 and len(guide.samples) > 24000

    engine = ApplioEngine(models)
    engine.load(voice)
    engine.load(voice)  # idempotent
    converted = engine.convert(guide, 0)
    assert converted.samples.dtype == np.float32 and np.isfinite(converted.samples).all()
    assert abs(len(converted.samples) / converted.sample_rate - len(guide.samples) / 24000) < 0.2

    wm = get_watermarker("audioseal")
    marked = wm.embed(converted, CYV_MESSAGE)
    prob, msg = wm.detect(marked)
    assert prob > 0.9 and msg == CYV_MESSAGE
    clean_prob, clean_msg = wm.detect(converted)
    assert clean_prob < 0.5 and clean_msg is None

    sf.write(tmp_path / "out.wav", marked.samples, marked.sample_rate)
