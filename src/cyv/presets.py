"""Output presets loaded from presets.toml; builds ffmpeg arguments from data."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

DEFAULT_PRESET = "master"
_BITRATE = re.compile(r"^\d{1,4}k$")
_ALLOWED_KEYS = {
    "description",
    "ext",
    "codec",
    "sample_rate",
    "channels",
    "bitrate",
    "vbr",
    "application",
}


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    ext: str
    codec: str
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: str | None = None
    vbr: bool | None = None
    application: str | None = None

    def ffmpeg_args(self) -> list[str]:
        """Output-side ffmpeg options (everything between -i and the output file)."""
        args = ["-vn", "-map_metadata", "-1", "-c:a", self.codec]
        if self.sample_rate:
            args += ["-ar", str(self.sample_rate)]
        if self.channels:
            args += ["-ac", str(self.channels)]
        if self.bitrate:
            args += ["-b:a", self.bitrate]
        if self.vbr is not None:
            args += ["-vbr", "on" if self.vbr else "off"]
        if self.application:
            args += ["-application", self.application]
        return args


def _parse(name: str, raw: object) -> Preset:
    if not isinstance(raw, dict):
        raise PresetError(f"preset {name!r}: expected a table")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise PresetError(f"preset {name!r}: unknown keys {sorted(unknown)}")
    ext, codec = raw.get("ext"), raw.get("codec")
    if not (isinstance(ext, str) and re.fullmatch(r"[a-z0-9]{2,4}", ext)):
        raise PresetError(f"preset {name!r}: bad ext")
    if not (isinstance(codec, str) and re.fullmatch(r"[a-z0-9_]+", codec)):
        raise PresetError(f"preset {name!r}: bad codec")
    bitrate = raw.get("bitrate")
    if bitrate is not None and not (isinstance(bitrate, str) and _BITRATE.match(bitrate)):
        raise PresetError(f"preset {name!r}: bad bitrate")
    for key in ("sample_rate", "channels"):
        v = raw.get(key)
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            raise PresetError(f"preset {name!r}: bad {key}")
    app = raw.get("application")
    if app is not None and app not in ("voip", "audio", "lowdelay"):
        raise PresetError(f"preset {name!r}: bad application")
    return Preset(
        name=name,
        description=str(raw.get("description", "")),
        ext=ext,
        codec=codec,
        sample_rate=raw.get("sample_rate"),
        channels=raw.get("channels"),
        bitrate=bitrate,
        vbr=raw.get("vbr"),
        application=app,
    )


@lru_cache(maxsize=1)
def load_presets() -> dict[str, Preset]:
    text = resources.files("cyv").joinpath("presets.toml").read_text("utf-8")
    return {name: _parse(name, raw) for name, raw in tomllib.loads(text).items()}


def get_preset(name: str) -> Preset:
    presets = load_presets()
    if name not in presets:
        raise PresetError(f"unknown preset {name!r}; choose one of: {', '.join(sorted(presets))}")
    return presets[name]
