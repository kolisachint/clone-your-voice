"""In-process Applio VoiceConverter wrapper (design §4.1, §4.3).

Why this is not just ``VoiceConverter.convert_audio``: that method reads and writes
files, rewrites any ``trained`` substring in the index path, and Applio resolves its
base models (rmvpe, contentvec) relative to the process cwd. So we

* link the shared base models from ``models_dir/applio`` into the Applio checkout
  (``rvc/models/predictors/rmvpe.pt``, ``rvc/models/embedders/contentvec``),
* run import/load/convert inside a lock-guarded ``chdir`` to the checkout (the daemon's
  own cwd is untouched outside those calls), and
* call the same building blocks ``convert_audio`` uses (``get_vc``, ``load_hubert``,
  ``vc.pipeline``) on in-memory audio, keeping HuBERT, the voice and the FAISS index warm.

This pins to Applio internals at ``APPLIO_COMMIT``; revisit when the pin moves.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from cyv import paths
from cyvd.interfaces import Audio

_INFER_SR = 16000
_MPS_MARKERS = ("mps", "metal", "coreml", "placeholder storage has not been allocated")

# Fixed inference parameters (Applio defaults). Voice packs do not tune these in v1.
INDEX_RATE = 0.75
PROTECT = 0.5
VOLUME_ENVELOPE = 1.0
F0_METHOD = "rmvpe"
EMBEDDER = "contentvec"

_cwd_lock = threading.RLock()


class EngineError(RuntimeError):
    """Setup problems: missing Applio checkout, base models or voice files."""


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    with _cwd_lock:
        old = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)


def applio_dir() -> Path:
    override = os.environ.get("CYV_APPLIO_DIR")
    return Path(override).expanduser() if override else paths.home() / "applio"


def _auto_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _IndexCache:
    """Stands in for ``faiss`` inside Applio's pipeline module.

    Applio re-reads the index (tens of MB) and reconstructs every vector on each call.
    We read once per file and hand back a proxy whose ``reconstruct_n`` is memoized.
    """

    def __init__(self, faiss_module: Any) -> None:
        self._faiss = faiss_module
        self._cache: dict[tuple[str, int], Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._faiss, name)

    def read_index(self, path: str) -> Any:
        st = os.stat(path)
        key = (path, st.st_mtime_ns)
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = _CachedIndex(self._faiss.read_index(path))
        return self._cache[key]


class _CachedIndex:
    def __init__(self, index: Any) -> None:
        self._index = index
        self._vectors: np.ndarray | None = None

    def reconstruct_n(self, start: int, n: int) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self._index.reconstruct_n(0, self._index.ntotal)
        return self._vectors[start : start + n]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._index, name)


class ApplioEngine:
    def __init__(self, models_dir: Path, device: str | None = None) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._models_dir = Path(models_dir)
        self._device = device or _auto_device()
        self._root = applio_dir()
        self._voice_key: tuple[str, int] | None = None
        self._index_path = ""
        self._converter: Any = None
        self._embedder_loaded = False

    @property
    def device(self) -> str:
        return self._device

    # -- setup -----------------------------------------------------------------

    def _link_base_models(self) -> None:
        base = self._models_dir / "applio"
        links = {
            self._root / "rvc/models/predictors/rmvpe.pt": base / "predictors/rmvpe.pt",
            self._root / "rvc/models/embedders/contentvec": base / "embedders/contentvec",
        }
        for link, target in links.items():
            if not target.exists():
                raise EngineError(f"base model missing: {target} (run `cyv setup`)")
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.is_dir():
                if any(link.iterdir()):
                    raise EngineError(f"refusing to replace non-empty {link}")
                link.rmdir()
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)

    def _ensure_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        if not (self._root / "rvc/infer/infer.py").is_file():
            raise EngineError(f"Applio checkout not found at {self._root} (run `cyv setup`)")
        self._link_base_models()
        os.environ.setdefault("APPLIO_CONFIG_DIR", str(paths.home() / "applio-config"))
        with _chdir(self._root):
            if str(self._root) not in sys.path:
                sys.path.insert(0, str(self._root))
            from rvc.configs.config import inference_config
            from rvc.infer import infer, pipeline

            config = inference_config()
            if config.device != self._device:
                config.device = self._device
                config.gpu_name = None
            pipeline.faiss = _IndexCache(pipeline.faiss)
            self._converter = infer.VoiceConverter(config=config)
        return self._converter

    def load(self, voice_dir: Path) -> None:
        voice_dir = Path(voice_dir)
        model, index, manifest = (
            voice_dir / "model.pth",
            voice_dir / "model.index",
            voice_dir / "voice.json",
        )
        for f in (model, index, manifest):
            if not f.is_file():
                raise EngineError(f"voice pack file missing: {f}")
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if (meta.get("engine"), meta.get("rvc_version")) != ("applio-rvc", "v2"):
            raise EngineError(f"unsupported voice pack engine/version in {manifest}")
        if meta.get("embedder") != EMBEDDER or meta.get("f0_method") != F0_METHOD:
            raise EngineError(f"unsupported embedder/f0_method in {manifest}")

        key = (str(model.resolve()), model.stat().st_mtime_ns)
        if key == self._voice_key:
            return
        conv = self._ensure_converter()
        with _chdir(self._root):
            conv.get_vc(str(model.resolve()), 0)
            if not self._embedder_loaded:
                conv.load_hubert(EMBEDDER)
                conv.last_embedder_model = EMBEDDER
                self._embedder_loaded = True
        self._index_path = str(index.resolve())
        self._voice_key = key

    # -- inference -------------------------------------------------------------

    def convert(self, audio: Audio, pitch: int) -> Audio:
        if self._voice_key is None:
            raise EngineError("no voice loaded")
        conv = self._converter
        x = _to_16k(audio)
        with _cwd_lock, _chdir(self._root):
            try:
                out = self._run(conv, x, pitch)
            except (RuntimeError, NotImplementedError, TypeError) as e:
                if conv.config.device != "mps" or not any(
                    m in str(e).lower() for m in _MPS_MARKERS
                ):
                    raise
                self._to_cpu(conv)
                out = self._run(conv, x, pitch)
        return Audio(np.ascontiguousarray(out, dtype=np.float32), int(conv.tgt_sr))

    def _run(self, conv: Any, x: np.ndarray, pitch: int) -> np.ndarray:
        return conv.vc.pipeline(
            model=conv.hubert_model,
            net_g=conv.net_g,
            sid=0,
            audio=x,
            pitch=pitch,
            f0_method=F0_METHOD,
            file_index=self._index_path,
            index_rate=INDEX_RATE,
            pitch_guidance=conv.use_f0,
            volume_envelope=VOLUME_ENVELOPE,
            version=conv.version,
            protect=PROTECT,
            f0_autotune=False,
            f0_autotune_strength=1,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
        )

    def _to_cpu(self, conv: Any) -> None:
        """Mirror Applio's own MPS-failure retry (rebuild the pipeline on CPU)."""
        conv.config.device = "cpu"
        conv.config.gpu_name = None
        for m in (conv.hubert_model, conv.net_g):
            m.to("cpu")
        conv.setup_vc_instance()
        self._device = "cpu"


def _to_16k(audio: Audio) -> np.ndarray:
    import soxr

    x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    if audio.sample_rate != _INFER_SR:
        x = soxr.resample(x, audio.sample_rate, _INFER_SR, quality="VHQ")
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak / 0.95 > 1:
        x = x / (peak / 0.95)
    return np.ascontiguousarray(x, dtype=np.float32)
