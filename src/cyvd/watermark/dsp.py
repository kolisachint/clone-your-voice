"""Dependency-free resampling and residual helpers for the watermarkers."""

from __future__ import annotations

import numpy as np


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited FFT resample of a finite 1-D signal (length becomes round(n*out/in))."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    n_out = round(len(x) * sr_out / sr_in)
    spec = np.fft.rfft(x.astype(np.float64))
    out_bins = n_out // 2 + 1
    new = np.zeros(out_bins, dtype=spec.dtype)
    keep = min(len(spec), out_bins)
    new[:keep] = spec[:keep]
    if n_out % 2 == 0 and keep == out_bins and len(spec) > out_bins:
        new[-1] = new[-1].real  # Nyquist bin must be real
    y = np.fft.irfft(new, n=n_out) * (n_out / len(x))
    return y.astype(np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def add_residual(
    x: np.ndarray, sr: int, residual_16k: np.ndarray, native_sr: int = 16000
) -> np.ndarray:
    """Upsample a watermark residual from ``native_sr`` and add it to ``x`` (bandwidth kept)."""
    res = fit_length(resample(residual_16k, native_sr, sr), len(x))
    return np.clip(x + res, -1.0, 1.0).astype(np.float32)


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.tolist():
        value = (value << 1) | int(b)
    return value


def int_to_bits(value: int, nbits: int) -> list[int]:
    return [(value >> (nbits - 1 - i)) & 1 for i in range(nbits)]
