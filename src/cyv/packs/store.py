"""Installed voices and the default-voice pointer."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from cyv import paths
from cyv.packs.manifest import NAME_RE, PackError, load_manifest


@dataclass(frozen=True)
class VoiceInfo:
    name: str
    dir: Path
    manifest: dict
    warnings: tuple[str, ...] = ()


def _check_name(name: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise PackError(f"invalid voice name {name!r}")


def list_voices() -> list[VoiceInfo]:
    root = paths.voices_dir()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.name.startswith(".") or not d.is_dir():
            continue
        try:
            out.append(VoiceInfo(d.name, d, load_manifest(d)))
        except PackError:
            continue
    return out


def _default_file() -> Path:
    return paths.home() / "default_voice"


def get_default() -> str | None:
    try:
        name = _default_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name if NAME_RE.fullmatch(name) and (paths.voices_dir() / name).is_dir() else None


def set_default(name: str) -> None:
    _check_name(name)
    if not (paths.voices_dir() / name).is_dir():
        raise PackError(f"no such voice {name!r}")
    paths.ensure_private_dir(paths.home())
    _default_file().write_text(name + "\n", encoding="utf-8")


def remove_voice(name: str) -> None:
    _check_name(name)
    d = paths.voices_dir() / name
    if not d.is_dir():
        raise PackError(f"no such voice {name!r}")
    shutil.rmtree(d)
    if get_default() is None and _default_file().exists():
        _default_file().unlink(missing_ok=True)
