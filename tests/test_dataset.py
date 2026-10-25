import json
import shutil
import wave
import zipfile

import numpy as np
import pytest

from cyv.cli import main as cli
from cyv.dataset import DatasetError, analyze_samples, find_audio_files, prepare_dataset

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
RATE = 24000


def speechy(seconds, amp=0.3, clip=False, seed=0):
    """Amplitude-modulated tone bursts with pauses, roughly speech-shaped."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    env = (np.sin(2 * np.pi * 0.7 * t) > -0.2).astype(float)
    x = env * amp * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    x += env * 0.02 * rng.standard_normal(n)
    if clip:
        x = np.clip(x * 6, -1, 1)
    return x


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def make_dir(tmp_path, total=90, files=3, **kw):
    d = tmp_path / "My Voice"
    d.mkdir()
    for i in range(files):
        write_wav(d / f"rec{i}.wav", speechy(total / files, seed=i, **kw))
    return d


def attest(_r):
    return "Test Owner", "This is my own voice, recorded by me."


def test_analyze_samples():
    clipped = analyze_samples(np.clip(speechy(5) * 8, -1, 1).astype(np.float32), RATE)
    clean = analyze_samples(speechy(5).astype(np.float32), RATE)
    assert clipped.clipping_ratio > 0.01 > clean.clipping_ratio
    assert 0 < clean.silence_ratio < 0.6
    quiet = analyze_samples((speechy(5) * 0.001).astype(np.float32), RATE)
    assert quiet.speech_rms_db < clean.speech_rms_db


def test_good_set(tmp_path):
    d = make_dir(tmp_path)
    out = tmp_path / "ds.zip"
    rep = prepare_dataset(d, out, name="myvoice", attest=attest)
    assert rep.seconds >= 60
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert sorted(n for n in names if n.startswith("wavs/")) == [
            "wavs/0001.wav",
            "wavs/0002.wav",
            "wavs/0003.wav",
        ]
        consent = json.loads(z.read("consent.json"))
        meta = json.loads(z.read("dataset_meta.json"))
        z.extract("wavs/0001.wav", tmp_path / "x")
    assert consent["attestation_version"] == 1
    assert consent["owner_name"] == "Test Owner"
    assert consent["created_at"].endswith("+00:00")
    assert meta["name"] == "myvoice" and meta["base_voice"] == "am_adam"
    assert meta["lang"] == "a" and meta["dataset_seconds"] >= 60
    with wave.open(str(tmp_path / "x" / "wavs" / "0001.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 1, 2)


def test_too_short_rejected(tmp_path):
    d = make_dir(tmp_path, total=15, files=1)
    with pytest.raises(DatasetError, match="usable speech"):
        prepare_dataset(d, tmp_path / "ds.zip", attest=attest)
    assert not (tmp_path / "ds.zip").exists()


def test_clipped_rejected(tmp_path):
    d = make_dir(tmp_path, clip=True)
    with pytest.raises(DatasetError, match="clipping"):
        prepare_dataset(d, tmp_path / "ds.zip", attest=attest)


def test_short_warns(tmp_path):
    d = make_dir(tmp_path, total=120)
    rep = prepare_dataset(d, tmp_path / "ds.zip", attest=attest)
    assert any("5 minutes" in w for w in rep.warnings)


def test_no_overwrite_and_bad_name(tmp_path):
    d = make_dir(tmp_path)
    out = tmp_path / "ds.zip"
    out.write_bytes(b"x")
    with pytest.raises(DatasetError, match="exists"):
        prepare_dataset(d, out, attest=attest)
    with pytest.raises(DatasetError, match="--name"):
        prepare_dataset(d, tmp_path / "o2.zip", name="Bad Name!", attest=attest)


def test_empty_attestation_rejected(tmp_path):
    d = make_dir(tmp_path)
    with pytest.raises(DatasetError, match="attestation"):
        prepare_dataset(d, tmp_path / "ds.zip", attest=lambda r: ("me", "ok"))


def test_symlinks_not_followed(tmp_path):
    d = make_dir(tmp_path, files=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    write_wav(outside / "secret.wav", speechy(5))
    (d / "link.wav").symlink_to(outside / "secret.wav")
    (d / "linkdir").symlink_to(outside, target_is_directory=True)
    files, skipped = find_audio_files(d)
    assert [f.name for f in files] == ["rec0.wav"]
    assert len(skipped) == 2


def test_cli_prepare(tmp_path, capsys):
    d = make_dir(tmp_path)
    out = tmp_path / "cli.zip"
    args = ["dataset", "prepare", str(d), "-o", str(out), "--owner", "A B"]
    assert cli.main([*args, "--attest", "my own voice, mine"]) == 0
    assert "usable duration" in capsys.readouterr().out
    assert zipfile.is_zipfile(out)
    assert cli.main([*args, "--attest", "my own voice, mine"]) == 1  # no overwrite


def test_cli_prepare_noninteractive_needs_attest(tmp_path, monkeypatch):
    d = make_dir(tmp_path)
    monkeypatch.setattr("sys.stdin", open("/dev/null"))  # noqa: SIM115
    assert cli.main(["dataset", "prepare", str(d), "-o", str(tmp_path / "z.zip")]) == 2
