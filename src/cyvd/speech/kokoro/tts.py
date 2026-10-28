"""Kokoro TTS with weights loaded from the shared models dir (never from the HF cache)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from cyv import paths
from cyvd.interfaces import Audio

SAMPLE_RATE = 24000
_VOICE_RE = re.compile(r"^[a-z]{2}_[a-z0-9]{1,24}$")


class TTSError(RuntimeError):
    pass


def _auto_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _use_system_espeak() -> None:
    """Let CYV_ESPEAK_LIBRARY override misaki's bundled espeakng-loader library.

    The bundled Linux wheel (espeak-ng 1.52) aborts at init on some hosts; a system
    libespeak-ng works. macOS normally needs no override.
    """
    lib = os.environ.get("CYV_ESPEAK_LIBRARY")
    if not lib:
        return
    import misaki.espeak  # noqa: F401  (sets the bundled library first; we override it)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(lib)
    EspeakWrapper.set_data_path(os.environ.get("CYV_ESPEAK_DATA") or None)


class KokoroTTS:
    def __init__(self, device: str | None = None, models_dir: Path | None = None) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._dir = Path(models_dir or paths.models_dir()) / "kokoro"
        self._device = device or _auto_device()
        self._model: Any = None
        self._pipelines: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._device

    def _load(self, lang: str) -> Any:
        from kokoro import KModel, KPipeline

        _use_system_espeak()
        if self._model is None:
            weights, config = self._dir / "kokoro-v1_0.pth", self._dir / "config.json"
            for f in (weights, config):
                if not f.is_file():
                    raise TTSError(f"Kokoro file missing: {f} (run `cyv setup`)")
            self._model = (
                KModel(repo_id="hexgrad/Kokoro-82M", config=str(config), model=str(weights))
                .to(self._device)
                .eval()
            )
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(
                lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=self._model
            )
        return self._pipelines[lang]

    def synth(self, text: str, voice: str, lang: str) -> Audio:
        if not _VOICE_RE.match(voice):
            raise TTSError(f"invalid voice name: {voice!r}")
        voice_file = self._dir / "voices" / f"{voice}.pt"
        if not voice_file.is_file():
            raise TTSError(f"Kokoro voice missing: {voice_file} (run `cyv setup`)")
        pipeline = self._load(lang.lower())
        chunks = [
            r.audio.detach().cpu().numpy().reshape(-1)
            for r in pipeline(text, voice=str(voice_file))
            if r.audio is not None
        ]
        if not chunks:
            raise TTSError("Kokoro produced no audio (empty or unpronounceable text)")
        return Audio(np.concatenate(chunks).astype(np.float32, copy=False), SAMPLE_RATE)
