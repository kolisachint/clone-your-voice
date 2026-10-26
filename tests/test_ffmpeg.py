"""Tests for src/cyv/ffmpeg.py (thin CLI, stdlib only, no torch)."""

import shutil
import subprocess

import pytest

from cyv import ffmpeg

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@needs_ffmpeg
def test_to_wav_from_wav_passthrough(tmp_path):
    src = tmp_path / "in.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(src),
        ],
        check=True,
    )
    out = ffmpeg.to_wav(src.read_bytes())
    assert ffmpeg._wav_data_bytes(out) == 32000


@needs_ffmpeg
def test_to_wav_from_mp4_needs_seekable_input(tmp_path):
    # Regression: Voice Memos .m4a decoded from a pipe yields a header-only
    # WAV (ffmpeg exits 0 with no samples). Must decode to real PCM.
    src = tmp_path / "in.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "aac",
            str(src),
        ],
        check=True,
    )
    blob = src.read_bytes()
    assert blob[4:8] == b"ftyp"
    out = ffmpeg.to_wav(blob)
    assert ffmpeg._wav_data_bytes(out) == 96000  # 1 s mono 48 kHz 16-bit


@needs_ffmpeg
def test_to_wav_empty_input_raises():
    with pytest.raises(ffmpeg.FfmpegError):
        ffmpeg.to_wav(b"not audio at all")


def test_wav_data_bytes_rejects_header_only():
    assert ffmpeg._wav_data_bytes(b"RIFF" + b"\x00" * 74) == 0
