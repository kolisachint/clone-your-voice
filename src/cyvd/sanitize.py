"""Rebuild an untrusted Applio checkpoint as a clean, tensor-only .pth.

Runs in the daemon venv in its own subprocess: ``python -I -m cyvd.sanitize IN OUT``.
Loading uses ``weights_only=True`` only; anything the restricted unpickler refuses fails the run.

Keys mirror what Applio writes (rvc/train/process/extract_model.py) and reads
(rvc/infer/infer.py: weight, config, f0, version, vocoder).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ALLOWED_KEYS = frozenset(
    {
        "weight",
        "config",
        "epoch",
        "step",
        "sr",
        "f0",
        "version",
        "creation_date",
        "model_hash",
        "dataset_length",
        "model_name",
        "author",
        "embedder_model",
        "speakers_id",
        "vocoder",
        "info",
    }
)
REQUIRED_KEYS = frozenset({"weight", "config", "sr", "version"})
_MAX_DEPTH = 8
_SCALARS = (str, int, float, bool, type(None))


class SanitizeError(Exception):
    pass


def _clean(value: Any, path: str, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise SanitizeError(f"{path}: nesting too deep")
    if isinstance(value, torch.Tensor):
        if value.is_sparse or value.layout != torch.strided:
            raise SanitizeError(f"{path}: unsupported tensor layout")
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, (list, tuple)):
        return [_clean(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise SanitizeError(f"{path}: non-string key {k!r}")
            out[k] = _clean(v, f"{path}.{k}", depth + 1)
        return out
    raise SanitizeError(f"{path}: disallowed type {type(value).__name__}")


def sanitize(src: Path, dst: Path) -> None:
    try:
        obj = torch.load(src, weights_only=True, map_location="cpu")
    except Exception as exc:
        raise SanitizeError(f"refused to load checkpoint safely: {exc}") from exc
    if not isinstance(obj, Mapping):
        raise SanitizeError("checkpoint is not a dict")
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        raise SanitizeError(f"checkpoint missing keys: {sorted(missing)}")
    clean = {k: _clean(v, k) for k, v in obj.items() if k in ALLOWED_KEYS}
    weight = clean["weight"]
    if not isinstance(weight, dict) or not weight:
        raise SanitizeError("weight must be a non-empty state dict")
    if not all(isinstance(v, torch.Tensor) for v in weight.values()):
        raise SanitizeError("weight must contain only tensors")
    torch.save(clean, dst)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m cyvd.sanitize IN OUT", file=sys.stderr)
        return 2
    try:
        sanitize(Path(args[0]), Path(args[1]))
    except SanitizeError as exc:
        print(f"sanitize: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
