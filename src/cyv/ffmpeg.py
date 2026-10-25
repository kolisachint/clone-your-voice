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


def run(
    args: Sequence[str], *, timeout: float | None = 600, stdin_data: bytes | None = None
) -> None:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            stdin=None if stdin_data is not None else subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
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
    """Transcode untrusted audio to 16-bit mono PCM WAV (no resampling).

    The input is fed on stdin with only the `pipe` protocol allowed, so a playlist or concat
    script inside it cannot make ffmpeg open local files or network URLs.
    """
    with tempfile.TemporaryDirectory(prefix="cyv-") as td:
        dst = Path(td, "out.wav")
        run(
            [
                "-protocol_whitelist",
                "pipe",
                "-i",
                "pipe:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            stdin_data=data,
        )
        return dst.read_bytes()
