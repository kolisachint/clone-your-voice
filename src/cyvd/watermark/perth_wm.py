"""Resemble Perth watermarker (optional; import only if ``resemble-perth`` is installed).

Perth's implicit watermark carries no payload, so ``message`` is ignored on embed and a
positive detection reports ``CYV_MESSAGE``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cyvd.interfaces import Audio
from cyvd.watermark.message import CYV_MESSAGE


class PerthWatermarker:
    name = "perth"

    def __init__(self) -> None:
        try:
            import perth
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("the 'perth' watermarker needs `pip install resemble-perth`") from e
        self._wm: Any = perth.PerthImplicitWatermarker()

    def embed(self, audio: Audio, message: int) -> Audio:  # noqa: ARG002
        x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        out = self._wm.apply_watermark(x, watermark=None, sample_rate=audio.sample_rate)
        return Audio(np.asarray(out, dtype=np.float32).reshape(-1), audio.sample_rate)

    def detect(self, audio: Audio) -> tuple[float, int | None]:
        x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        score = self._wm.get_watermark(x, sample_rate=audio.sample_rate)
        prob = float(np.asarray(score).reshape(-1)[0]) if score is not None else 0.0
        return prob, (CYV_MESSAGE if prob >= 0.5 else None)
