"""Meta AudioSeal (16-bit) watermarker, run at its native 16 kHz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cyv import paths
from cyvd.interfaces import Audio
from cyvd.watermark import dsp
from cyvd.watermark.message import NBITS, unpack

NATIVE_SR = 16000
DETECT_THRESHOLD = 0.5


class AudioSealWatermarker:
    name = "audioseal"

    def __init__(self, models_dir: Path | None = None, device: str = "cpu") -> None:
        self._dir = Path(models_dir or paths.models_dir()) / "audioseal"
        self._device = device
        self._gen: Any = None
        self._det: Any = None

    def _load(self) -> None:
        if self._gen is not None:
            return
        from audioseal import AudioSeal

        gen, det = self._dir / "generator_base.pth", self._dir / "detector_base.pth"
        for f in (gen, det):
            if not f.is_file():
                raise FileNotFoundError(f"AudioSeal checkpoint missing: {f} (run `cyv setup`)")
        self._gen = AudioSeal.load_generator(str(gen), nbits=NBITS).to(self._device).eval()
        self._det = AudioSeal.load_detector(str(det), nbits=NBITS).to(self._device).eval()

    def embed(self, audio: Audio, message: int) -> Audio:
        import torch

        self._load()
        x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        x16 = dsp.resample(x, audio.sample_rate, NATIVE_SR)
        msg = torch.tensor(
            [dsp.int_to_bits(message, NBITS)], dtype=torch.int64, device=self._device
        )
        with torch.inference_mode():
            wav = torch.from_numpy(x16).to(self._device)[None, None, :]
            residual = self._gen.get_watermark(wav, NATIVE_SR, message=msg)
        res = residual[0, 0].cpu().numpy()
        out = dsp.add_residual(x, audio.sample_rate, dsp.fit_length(res, len(x16)), NATIVE_SR)
        return Audio(out, audio.sample_rate)

    def detect(self, audio: Audio) -> tuple[float, int | None]:
        import torch

        self._load()
        x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        x16 = dsp.resample(x, audio.sample_rate, NATIVE_SR)
        with torch.inference_mode():
            wav = torch.from_numpy(x16).to(self._device)[None, None, :]
            if self._det.normalizer is not None:
                wav = self._det.normalizer.loudness_normalization(wav)
            out = self._det.detector(wav)  # (1, 2 + nbits, frames) raw logits
        raw = out[:, 2:, :]
        positive = torch.softmax(out[0, :2], dim=0)[1] > DETECT_THRESHOLD
        prob = float(positive.float().mean())
        if prob < DETECT_THRESHOLD:
            return prob, None
        # Decode from watermarked frames only, so partly edited clips still read cleanly.
        logits = raw[0][:, positive].mean(dim=-1)
        bits = (logits > 0).int().cpu().numpy()
        return prob, dsp.bits_to_int(bits)


def message_is_cyv(message: int | None) -> bool:
    return message is not None and unpack(message)[2]
