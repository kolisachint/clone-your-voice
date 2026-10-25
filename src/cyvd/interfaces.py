"""Internal component boundaries (design §4.6). Not a public plug-in API yet.

Audio is always passed as (samples, sample_rate) where samples is a 1-D float32
numpy array in [-1, 1], mono.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32, shape (n,)
    sample_rate: int


class TTS(Protocol):
    def synth(self, text: str, voice: str, lang: str) -> Audio: ...


class Engine(Protocol):
    def load(self, voice_dir: Path) -> None:
        """Load a sanitized, installed voice pack directory. Idempotent for the same dir."""

    def convert(self, audio: Audio, pitch: int) -> Audio: ...

    @property
    def device(self) -> str: ...


class Watermarker(Protocol):
    name: str

    def embed(self, audio: Audio, message: int) -> Audio:
        """Embed a 16-bit message. Never encode anything identifying the user."""

    def detect(self, audio: Audio) -> tuple[float, int | None]:
        """Return (probability watermarked, decoded 16-bit message or None)."""
