"""Mono 16-bit PCM WAV encode/decode with stdlib `wave` + numpy."""

from __future__ import annotations

import io
import wave

import numpy as np

from cyvd.interfaces import Audio


class WavError(ValueError):
    pass


def encode_wav(audio: Audio) -> bytes:
    x = np.clip(np.asarray(audio.samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_wav(data: bytes) -> Audio:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate, n = (
                w.getnchannels(),
                w.getsampwidth(),
                w.getframerate(),
                w.getnframes(),
            )
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise WavError(f"invalid WAV: {e}") from e
    if width != 2:
        raise WavError("only 16-bit PCM WAV is supported")
    if rate <= 0 or channels <= 0:
        raise WavError("invalid WAV header")
    if n == 0 or not raw:
        raise WavError("audio is empty")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    if x.size == 0:
        raise WavError("audio is empty")
    return Audio(np.ascontiguousarray(x, dtype=np.float32), rate)


def wav_sample_rate(data: bytes) -> int:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getframerate()
