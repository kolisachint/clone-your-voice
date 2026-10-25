"""Voice pack import (design §6.1): zip hygiene, validation, sha256, sanitize, atomic install."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from cyv import paths
from cyv.packs.manifest import NAME_RE, PackError, load_manifest, validate_consent
from cyv.packs.store import VoiceInfo

ALLOWED_MEMBERS = ("model.pth", "model.index", "voice.json", "consent.json")
MAX_TOTAL = 1 << 30
MAX_MEMBER = {
    "model.pth": 1 << 30,
    "model.index": 1 << 30,
    "voice.json": 1 << 20,
    "consent.json": 1 << 20,
}
SANITIZE_TIMEOUT = 300
_CHUNK = 1 << 20
STALE_TMP_SECONDS = 3600
_PIN_RE = re.compile(r"^APPLIO_COMMIT\s*=\s*[\"']([0-9a-f]{7,40})[\"']", re.M)


def _pinned_commit() -> str | None:
    """Read APPLIO_COMMIT from cyvd's pin.py as text (importing cyvd.engine_applio pulls in the
    engine). Returns None if cyvd is not installed next to cyv."""
    try:
        spec = importlib.util.find_spec("cyvd")  # runs only cyvd/__init__.py
        roots = list(spec.submodule_search_locations or []) if spec else []
        for root in roots:
            pin = Path(root) / "engine_applio" / "pin.py"
            if pin.is_file():
                m = _PIN_RE.search(pin.read_text(encoding="utf-8"))
                return m.group(1) if m else None
    except (ImportError, ValueError, OSError):
        pass
    return None


def _clean_stale(voices: Path) -> None:
    cutoff = time.time() - STALE_TMP_SECONDS
    for d in voices.glob(".import-*"):
        try:
            if d.is_dir() and not d.is_symlink() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def _check_infos(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = zf.infolist()
    seen: set[str] = set()
    total = 0
    for info in infos:
        name = info.filename
        if name in seen:
            raise PackError(f"duplicate member {name!r}")
        seen.add(name)
        if name.startswith("/") or "\\" in name or ".." in name.split("/") or ":" in name:
            raise PackError(f"unsafe member path {name!r}")
        if info.is_dir() or name.endswith("/"):
            raise PackError(f"directories are not allowed ({name!r})")
        if name not in ALLOWED_MEMBERS:
            raise PackError(f"unexpected member {name!r}")
        mode = info.external_attr >> 16
        if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise PackError(f"member {name!r} is not a regular file (symlink?)")
        if info.file_size > MAX_MEMBER[name]:
            raise PackError(f"member {name!r} too large")
        total += info.file_size
    missing = [m for m in ALLOWED_MEMBERS if m not in seen]
    if missing:
        raise PackError(f"pack is missing {', '.join(missing)}")
    if total > MAX_TOTAL:
        raise PackError("pack too large when decompressed")
    return infos


def _extract(zf: zipfile.ZipFile, dest: Path) -> None:
    total = 0
    for info in _check_infos(zf):
        cap = MAX_MEMBER[info.filename]
        written = 0
        # Names are allowlisted, so dest / name cannot escape dest.
        with zf.open(info) as src, open(dest / info.filename, "wb") as out:
            while chunk := src.read(_CHUNK):
                written += len(chunk)
                total += len(chunk)
                if written > cap or total > MAX_TOTAL:
                    raise PackError(f"member {info.filename!r} exceeds size limit")
                out.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _sanitize(tmp: Path, daemon_python: Path | None) -> None:
    """Rewrite tmp/model.pth and tmp/model.index in place via the sandboxed sanitizer."""
    py = Path(daemon_python) if daemon_python else paths.venv_dir() / "bin" / "python"
    if not py.exists():
        raise PackError(f"daemon python not found at {py}; run `cyv setup` first")
    pth_out, idx_out = tmp / "model.sanitized", tmp / "model.index.sanitized"
    try:
        proc = subprocess.run(
            [
                str(py),
                "-I",
                "-m",
                "cyvd.sanitize",
                str(tmp / "model.pth"),
                str(pth_out),
                "--manifest",
                str(tmp / "voice.json"),
                "--index",
                str(tmp / "model.index"),
                "--index-out",
                str(idx_out),
            ],  # fmt: skip
            capture_output=True,
            text=True,
            timeout=SANITIZE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PackError("sanitizing model.pth timed out") from exc
    if proc.returncode != 0 or not pth_out.is_file() or not idx_out.is_file():
        msg = (proc.stderr or "").strip().splitlines()
        raise PackError(f"model rejected: {msg[-1] if msg else 'sanitizer failed'}")
    os.replace(pth_out, tmp / "model.pth")
    os.replace(idx_out, tmp / "model.index")


def import_pack(path: Path, *, daemon_python: Path | None = None, force: bool = False) -> VoiceInfo:
    path = Path(path)
    voices = paths.ensure_private_dir(paths.voices_dir())
    _clean_stale(voices)
    tmp = Path(tempfile.mkdtemp(prefix=".import-", dir=voices))
    try:
        try:
            with zipfile.ZipFile(path) as zf:
                _extract(zf, tmp)
        except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError) as exc:
            raise PackError(f"cannot read pack {path.name}: {exc}") from exc
        manifest = load_manifest(tmp)
        try:
            validate_consent(json.loads((tmp / "consent.json").read_text(encoding="utf-8")))
        except (ValueError, RecursionError) as exc:
            raise PackError(f"consent.json is not valid JSON: {exc}") from exc
        name = manifest["name"]
        if not NAME_RE.fullmatch(name):
            raise PackError(f"invalid voice name {name!r}")
        source = {}
        for member, want in manifest["sha256"].items():
            source[member] = _sha256(tmp / member)
            if source[member] != want:
                raise PackError(f"sha256 mismatch for {member}")
        warnings: list[str] = []
        pin = _pinned_commit()
        ev = manifest["engine_version"]
        if pin and not (pin.startswith(ev) or ev.startswith(pin)):
            warnings.append(
                f"pack was trained with Applio {ev[:12]}, this cyv uses {pin[:12]}; "
                "the voice may sound different"
            )
        target = voices / name
        replacing = target.exists()
        if replacing and not force:
            raise PackError(f"voice {name!r} already exists (use --force to replace)")
        _sanitize(tmp, daemon_python)
        installed = {
            "sanitized_sha256": {m: _sha256(tmp / m) for m in ("model.pth", "model.index")},
            "source_sha256": source,
            "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (tmp / "installed.json").write_text(
            json.dumps(installed, indent=2) + "\n", encoding="utf-8"
        )
        for f in tmp.iterdir():
            os.chmod(f, 0o600)
        os.chmod(tmp, 0o700)
        try:
            if replacing:
                old = voices / f".old-{name}-{os.getpid()}"
                os.rename(target, old)
                try:
                    os.rename(tmp, target)
                except OSError:
                    os.rename(old, target)
                    raise
                shutil.rmtree(old, ignore_errors=True)
            else:
                os.rename(tmp, target)
        except OSError as exc:
            raise PackError(f"cannot install voice {name!r}: {exc}") from exc
        return VoiceInfo(name=name, dir=target, manifest=manifest, warnings=tuple(warnings))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    print(import_pack(Path(sys.argv[1])))
