"""Thin ffmpeg wrapper: argument lists only, never a shell, private temp dirs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cyv.presets import Preset

INSTALL_HINT = "install ffmpeg with: brew install ffmpeg"


class FfmpegError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FfmpegError(f"ffmpeg not found on PATH; {INSTALL_HINT}")
    return exe


def version() -> str | None:
    """First line of `ffmpeg -version`, or None if unavailable."""
    try:
        out = subprocess.run(
            [find_ffmpeg(), "-version"], capture_output=True, text=True, timeout=15, check=True
        )
    except (FfmpegError, OSError, subprocess.SubprocessError):
        return None
    return out.stdout.splitlines()[0] if out.stdout else None


def run(args: Sequence[str], *, timeout: float | None = 600) -> None:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise FfmpegError("ffmpeg timed out") from e
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FfmpegError(f"ffmpeg failed (exit {proc.returncode}): {msg[-1] if msg else ''}")


def encode(wav: bytes, preset: Preset) -> bytes:
    """Encode WAV bytes with a preset and return the encoded file's bytes."""
    with tempfile.TemporaryDirectory(prefix="cyv-") as td:
        src, dst = Path(td, "in.wav"), Path(td, f"out.{preset.ext}")
        src.write_bytes(wav)
        run(["-i", str(src), *preset.ffmpeg_args(), str(dst)])
        return dst.read_bytes()


def to_wav(data: bytes) -> bytes:
    """Transcode any ffmpeg-readable audio to 16-bit PCM WAV (no resampling)."""
    with tempfile.TemporaryDirectory(prefix="cyv-") as td:
        src, dst = Path(td, "in.bin"), Path(td, "out.wav")
        src.write_bytes(data)
        run(["-i", str(src), "-vn", "-c:a", "pcm_s16le", str(dst)])
        return dst.read_bytes()
