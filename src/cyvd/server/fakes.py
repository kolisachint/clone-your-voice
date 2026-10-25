"""Model-free stand-ins used by `--fake` (tests, CI)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from cyvd.interfaces import Audio

FAKE_SR = 24000


def _delay() -> None:
    d = float(os.environ.get("CYVD_FAKE_DELAY", "0") or 0)
    if d > 0:
        time.sleep(d)


class FakeTTS:
    def synth(self, text: str, voice: str, lang: str) -> Audio:
        _delay()
        secs = min(0.05 * max(len(text), 1), 5.0)
        t = np.arange(int(FAKE_SR * secs)) / FAKE_SR
        return Audio((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), FAKE_SR)


class FakeEngine:
    device = "fake"

    def __init__(self) -> None:
        self.voice_dir: Path | None = None

    def load(self, voice_dir: Path) -> None:
        self.voice_dir = voice_dir

    def convert(self, audio: Audio, pitch: int) -> Audio:
        _delay()
        return audio


class FakeWatermarker:
    name = "fake"

    def embed(self, audio: Audio, message: int) -> Audio:
        return audio

    def detect(self, audio: Audio) -> tuple[float, int | None]:
        return 0.0, None
