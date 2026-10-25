"""Rebuild an untrusted Applio checkpoint as a clean, tensor-only .pth.

Runs in the daemon venv in its own subprocess:
``python -I -m cyvd.sanitize IN OUT --manifest voice.json --index IN.index --index-out OUT.index``.
Loading uses ``weights_only=True`` only; anything the restricted unpickler refuses fails the run.
The checkpoint is cross-checked against voice.json, and model.index is parsed with faiss
(allowlisted index types, d == 768, size caps) and re-written fresh.

Test-only switch: ``CYV_SANITIZE_SKIP_INDEX=1`` skips faiss validation and copies the index
bytes unchanged. It exists so unit tests can run in a venv without faiss. Never set it for real
imports.

Keys mirror what Applio writes (rvc/train/process/extract_model.py) and reads
(rvc/infer/infer.py: weight, config, f0, version, vocoder).
"""

from __future__ import annotations

import json
import os
import re
import shutil
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
MIN_TORCH = (2, 6)  # older torch has weights_only=True unpickler bypasses (RCE)
INDEX_DIM = 768
MAX_INDEX_FILE = 1 << 30
MAX_INDEX_BYTES = 2 << 30  # ntotal * d * 4
INDEX_TYPES = frozenset({"IndexFlatL2", "IndexFlatIP", "IndexFlat", "IndexIVFFlat"})
_SKIP_ENV = "CYV_SANITIZE_SKIP_INDEX"


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


def _check_torch_version() -> None:
    m = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    if not m or (int(m.group(1)), int(m.group(2))) < MIN_TORCH:
        raise SanitizeError(
            f"torch {torch.__version__} is too old for safe checkpoint loading (need >= 2.6)"
        )


def _norm_sr(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.fullmatch(r"(\d+)k?", v.strip().lower())
        if m:
            n = int(m.group(1))
            return n * 1000 if v.strip().lower().endswith("k") else n
    return None


def _cross_check(clean: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    rv = manifest.get("rvc_version")
    if clean["version"] != "v2" or rv != "v2":
        raise SanitizeError(
            f"checkpoint version {clean['version']!r} / voice.json rvc_version {rv!r}: need v2"
        )
    sr = _norm_sr(clean["sr"])
    if sr is None or sr != manifest.get("sample_rate"):
        raise SanitizeError(
            f"checkpoint sr {clean['sr']!r} does not match voice.json sample_rate "
            f"{manifest.get('sample_rate')!r}"
        )
    if not clean.get("f0"):
        raise SanitizeError("checkpoint has no pitch (f0) model, but rmvpe needs one")
    if "vocoder" in clean and clean["vocoder"] != manifest.get("vocoder"):
        raise SanitizeError(
            f"checkpoint vocoder {clean['vocoder']!r} does not match voice.json "
            f"{manifest.get('vocoder')!r}"
        )
    if "embedder_model" in clean and clean["embedder_model"] != manifest.get("embedder"):
        raise SanitizeError(
            f"checkpoint embedder {clean['embedder_model']!r} does not match voice.json "
            f"{manifest.get('embedder')!r}"
        )


def sanitize_index(src: Path, dst: Path) -> None:
    try:
        size = src.stat().st_size
    except OSError as exc:
        raise SanitizeError(f"cannot read index: {exc}") from exc
    if size > MAX_INDEX_FILE:
        raise SanitizeError("index file too large")
    if os.environ.get(_SKIP_ENV) == "1":  # test-only
        shutil.copyfile(src, dst)
        return
    try:
        import faiss
    except ImportError as exc:
        raise SanitizeError(
            "faiss is not installed in the daemon venv; cannot validate model.index"
        ) from exc
    try:
        index = faiss.read_index(str(src))
    except Exception as exc:
        raise SanitizeError(f"model.index is not a readable faiss index: {exc}") from exc
    kind = type(index).__name__
    if kind not in INDEX_TYPES:
        raise SanitizeError(f"index type {kind} is not allowed")
    if index.d != INDEX_DIM:
        raise SanitizeError(f"index dimension {index.d} != {INDEX_DIM}")
    if index.ntotal * index.d * 4 > MAX_INDEX_BYTES:
        raise SanitizeError("index holds too many vectors")
    faiss.write_index(index, str(dst))


def sanitize(
    src: Path,
    dst: Path,
    manifest: Mapping[str, Any] | None = None,
    index: tuple[Path, Path] | None = None,
) -> None:
    _check_torch_version()
    if index is not None:
        sanitize_index(*index)
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
    if manifest is not None:
        _cross_check(clean, manifest)
    torch.save(clean, dst)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    usage = (
        "usage: python -m cyvd.sanitize IN OUT [--manifest voice.json] "
        "[--index IN.index --index-out OUT.index]"
    )
    opts: dict[str, str] = {}
    pos: list[str] = []
    it = iter(args)
    for a in it:
        if a in ("--manifest", "--index", "--index-out"):
            v = next(it, None)
            if v is None:
                print(usage, file=sys.stderr)
                return 2
            opts[a] = v
        else:
            pos.append(a)
    if len(pos) != 2 or ("--index" in opts) != ("--index-out" in opts):
        print(usage, file=sys.stderr)
        return 2
    try:
        manifest = None
        if "--manifest" in opts:
            try:
                manifest = json.loads(Path(opts["--manifest"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError) as exc:
                raise SanitizeError(f"cannot read manifest: {exc}") from exc
            if not isinstance(manifest, dict):
                raise SanitizeError("manifest must be a JSON object")
        index = (Path(opts["--index"]), Path(opts["--index-out"])) if "--index" in opts else None
        sanitize(Path(pos[0]), Path(pos[1]), manifest, index)
    except SanitizeError as exc:
        print(f"sanitize: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
