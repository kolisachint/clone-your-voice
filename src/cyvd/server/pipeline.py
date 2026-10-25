"""Request-level pipeline: voice resolution, TTS/convert, watermark, WAV encode."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyv import paths
from cyvd.interfaces import TTS, Audio, Engine, Watermarker
from cyvd.server.wavio import decode_wav, encode_wav

VOICE_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


class RequestError(Exception):
    """An error with a protocol `code`, reported to the client as ok=false."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VoiceConfig:
    name: str
    directory: Path
    tts_voice: str
    tts_lang: str
    pitch_shift: int


def validate_voice_name(name: object) -> str:
    if not isinstance(name, str) or not VOICE_RE.fullmatch(name):
        raise RequestError("bad_voice", "voice name must match ^[a-z0-9_-]{1,32}$")
    return name


class Pipeline:
    def __init__(
        self,
        tts: TTS,
        engine: Engine,
        watermarker: Watermarker,
        message: int,
        *,
        lenient_voices: bool = False,
    ) -> None:
        self.tts = tts
        self.engine = engine
        self.watermarker = watermarker
        self.message = message
        self.lenient = lenient_voices
        self.loaded: str | None = None

    @property
    def device(self) -> str:
        return self.engine.device

    def resolve_voice(self, name: str | None) -> VoiceConfig:
        if name is None:
            f = paths.home() / "default_voice"
            try:
                name = f.read_text().strip()
            except OSError:
                if not self.lenient:
                    raise RequestError(
                        "no_voice", "no voice given and no default voice set"
                    ) from None
                name = "fake"
        validate_voice_name(name)
        d = paths.voices_dir() / name
        meta: dict[str, Any] = {}
        try:
            meta = json.loads((d / "voice.json").read_text())
        except (OSError, ValueError):
            if not self.lenient:
                raise RequestError("voice_not_found", f"voice '{name}' is not installed") from None
        base = meta.get("tts_base") or {}
        return VoiceConfig(
            name=name,
            directory=d,
            tts_voice=str(base.get("voice", "af_heart")),
            tts_lang=str(base.get("lang", "en-us")),
            pitch_shift=int(meta.get("pitch_shift", 0)),
        )

    def load(self, cfg: VoiceConfig) -> None:
        if self.loaded != cfg.name:
            self.engine.load(cfg.directory)
            self.loaded = cfg.name

    def _finish(self, audio: Audio, watermark: bool) -> bytes:
        if watermark:
            audio = self.watermarker.embed(audio, self.message)
        return encode_wav(audio)

    def say(self, text: str, voice: str | None, pitch: int, watermark: bool) -> bytes:
        cfg = self.resolve_voice(voice)
        self.load(cfg)
        guide = self.tts.synth(text, cfg.tts_voice, cfg.tts_lang)
        out = self.engine.convert(guide, pitch + cfg.pitch_shift)
        return self._finish(out, watermark)

    def convert(self, wav: bytes, voice: str | None, pitch: int, watermark: bool) -> bytes:
        cfg = self.resolve_voice(voice)
        self.load(cfg)
        out = self.engine.convert(decode_wav(wav), pitch + cfg.pitch_shift)
        return self._finish(out, watermark)

    def reload(self, voice: str | None) -> None:
        cfg = self.resolve_voice(voice)
        self.loaded = None
        self.load(cfg)


def build_pipeline(fake: bool) -> Pipeline:
    if fake:
        from cyvd.server.fakes import FakeEngine, FakeTTS, FakeWatermarker

        return Pipeline(FakeTTS(), FakeEngine(), FakeWatermarker(), 0, lenient_voices=True)

    from cyvd.engine_applio import ApplioEngine
    from cyvd.tts_kokoro import KokoroTTS
    from cyvd.watermark import CYV_MESSAGE, get_watermarker

    return Pipeline(
        KokoroTTS(),
        ApplioEngine(paths.models_dir()),
        get_watermarker("audioseal"),
        CYV_MESSAGE,
    )
