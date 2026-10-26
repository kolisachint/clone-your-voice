"""Thin ffmpeg wrapper: argument lists only, never a shell, private temp dirs."""

from __future__ import annotations

import shutil
import struct
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


def _wav_data_bytes(wav: bytes) -> int:
    """Size of the PCM payload, or 0 if the bytes are not a playable WAV."""
    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return 0
    i = wav.find(b"data", 12)
    if i < 0 or i + 8 > len(wav):
        return 0
    (size,) = struct.unpack_from("<I", wav, i + 4)
    return size if i + 8 + size <= len(wav) else 0


def _is_mp4(data: bytes) -> bool:
    """True for MP4-family containers (m4a/mp4/mov): `ftyp` box at offset 4."""
    return len(data) > 8 and data[4:8] == b"ftyp"


def to_wav(data: bytes) -> bytes:
    """Transcode untrusted audio to 16-bit mono PCM WAV (no resampling).

    Streamable formats are fed on stdin with only the `pipe` protocol allowed,
    so a playlist or concat script inside them cannot make ffmpeg open local
    files or network URLs. MP4-family input (e.g. Voice Memos `.m4a`) needs a
    seekable input — from a pipe the demuxer fails and ffmpeg exits 0 with an
    empty file — so it is decoded from a temp file instead. The temp file is
    only used when the bytes carry the `ftyp` magic, never for playlists, and
    `http`/`https` stay off the protocol whitelist either way."""
    with tempfile.TemporaryDirectory(prefix="cyv-") as td:
        dst = Path(td, "out.wav")
        if _is_mp4(data):
            src = Path(td, "in.m4a")
            src.write_bytes(data)
            run(
                [
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(src),
                    "-vn",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(dst),
                ]
            )
        else:
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
        out = dst.read_bytes()
    if _wav_data_bytes(out) == 0:
        raise FfmpegError("ffmpeg decoded no audio from the input")
    return out
