"""Hand-written validator mirroring voice_schema.json (no jsonschema dependency)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class PackError(Exception):
    """Raised for any invalid, unsafe or unsupported voice pack."""


NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_VOICE_RE = re.compile(r"^[a-z]{2}_[a-z0-9]{1,24}$")
_LANG_RE = re.compile(r"^[a-z]$")

_REQUIRED = (
    "schema_version", "name", "engine", "engine_version", "rvc_version", "sample_rate",
    "embedder", "f0_method", "vocoder", "tts_base", "pitch_shift", "sha256",
)  # fmt: skip
_OPTIONAL = ("epochs", "dataset_seconds", "license")
_ENUMS: dict[str, tuple[Any, ...]] = {
    "engine": ("applio-rvc",),
    "rvc_version": ("v2",),
    "sample_rate": (32000, 40000, 48000),
    "embedder": ("contentvec",),
    "f0_method": ("rmvpe",),
    "vocoder": ("HiFi-GAN",),
}
CONSENT_KEYS = ("attestation", "attestation_version", "owner_name", "created_at")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pattern(obj: dict, key: str, rx: re.Pattern[str], where: str) -> None:
    v = obj[key]
    if not isinstance(v, str) or not rx.fullmatch(v):
        raise PackError(f"{where}{key}: invalid value {v!r}")


def _exact_keys(
    obj: dict, required: tuple[str, ...], optional: tuple[str, ...], where: str
) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise PackError(f"{where or 'voice.json'}: missing field(s) {', '.join(missing)}")
    extra = sorted(set(obj) - set(required) - set(optional))
    if extra:
        raise PackError(f"{where or 'voice.json'}: unknown field(s) {', '.join(extra)}")


def validate_manifest(obj: dict) -> None:
    if not isinstance(obj, dict):
        raise PackError("voice.json must be a JSON object")
    _exact_keys(obj, _REQUIRED, _OPTIONAL, "")
    sv = obj["schema_version"]
    if not _is_int(sv) or sv != 1:
        raise PackError(f"unsupported schema_version {sv!r} (this cyv supports 1)")
    _pattern(obj, "name", NAME_RE, "")
    for key, allowed in _ENUMS.items():
        v = obj[key]
        if not any(type(v) is type(a) and v == a for a in allowed):
            raise PackError(f"unsupported {key} {v!r} (supported: {', '.join(map(str, allowed))})")
    _pattern(obj, "engine_version", _COMMIT_RE, "")
    if "epochs" in obj and not (_is_int(obj["epochs"]) and obj["epochs"] >= 1):
        raise PackError("epochs must be an integer >= 1")
    if "dataset_seconds" in obj and not (
        _is_num(obj["dataset_seconds"]) and obj["dataset_seconds"] >= 0
    ):
        raise PackError("dataset_seconds must be a number >= 0")
    if "license" in obj and not isinstance(obj["license"], str):
        raise PackError("license must be a string")
    ps = obj["pitch_shift"]
    if not (_is_int(ps) and -24 <= ps <= 24):
        raise PackError("pitch_shift must be an integer in [-24, 24]")
    tb = obj["tts_base"]
    if not isinstance(tb, dict):
        raise PackError("tts_base must be an object")
    _exact_keys(tb, ("voice", "lang"), (), "tts_base")
    _pattern(tb, "voice", _VOICE_RE, "tts_base.")
    _pattern(tb, "lang", _LANG_RE, "tts_base.")
    sha = obj["sha256"]
    if not isinstance(sha, dict):
        raise PackError("sha256 must be an object")
    _exact_keys(sha, ("model.pth", "model.index"), (), "sha256")
    for k in sha:
        _pattern(sha, k, _SHA_RE, "sha256.")


def validate_consent(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise PackError("consent.json must be a JSON object")
    _exact_keys(obj, CONSENT_KEYS, (), "consent.json")
    for k in ("attestation", "owner_name", "created_at"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            raise PackError(f"consent.json: {k} must be a non-empty string")
    v = obj["attestation_version"]
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise PackError("consent.json: attestation_version must be a positive integer")


def load_manifest(voice_dir: Path) -> dict:
    path = Path(voice_dir) / "voice.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise PackError(f"cannot read {path}: {exc}") from exc
    try:
        validate_manifest(obj)
    except RecursionError as exc:
        raise PackError("voice.json is nested too deeply") from exc
    return obj
