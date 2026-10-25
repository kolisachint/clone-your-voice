"""Audio watermarking (design §6.3, ADR-004). Backends are imported lazily."""

from __future__ import annotations

from cyvd.interfaces import Watermarker
from cyvd.watermark.message import CYV_MESSAGE, FORMAT_VERSION, pack, unpack

__all__ = ["CYV_MESSAGE", "FORMAT_VERSION", "get_watermarker", "pack", "unpack"]


def get_watermarker(name: str = "audioseal") -> Watermarker:
    if name == "audioseal":
        from cyvd.watermark.audioseal_wm import AudioSealWatermarker

        return AudioSealWatermarker()
    if name == "perth":
        from cyvd.watermark.perth_wm import PerthWatermarker

        return PerthWatermarker()
    raise ValueError(f"unknown watermarker {name!r} (expected 'audioseal' or 'perth')")
