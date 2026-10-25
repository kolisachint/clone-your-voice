"""Voice pack import (design §6.1): zip hygiene, validation, sha256, sanitize, atomic install."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
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


def _sanitize(src: Path, dst: Path, daemon_python: Path | None) -> None:
    py = Path(daemon_python) if daemon_python else paths.venv_dir() / "bin" / "python"
    if not py.exists():
        raise PackError(f"daemon python not found at {py}; run `cyv setup` first")
    try:
        proc = subprocess.run(
            [str(py), "-I", "-m", "cyvd.sanitize", str(src), str(dst)],
            capture_output=True,
            text=True,
            timeout=SANITIZE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PackError("sanitizing model.pth timed out") from exc
    if proc.returncode != 0 or not dst.is_file():
        msg = (proc.stderr or "").strip().splitlines()
        raise PackError(f"model.pth rejected: {msg[-1] if msg else 'sanitizer failed'}")


def import_pack(path: Path, *, daemon_python: Path | None = None, force: bool = False) -> VoiceInfo:
    path = Path(path)
    voices = paths.ensure_private_dir(paths.voices_dir())
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
        except ValueError as exc:
            raise PackError(f"consent.json is not valid JSON: {exc}") from exc
        name = manifest["name"]
        if not NAME_RE.match(name):
            raise PackError(f"invalid voice name {name!r}")
        for member, want in manifest["sha256"].items():
            if _sha256(tmp / member) != want:
                raise PackError(f"sha256 mismatch for {member}")
        target = voices / name
        if target.exists() and not force:
            raise PackError(f"voice {name!r} already exists (use --force to replace)")
        clean = tmp / "model.sanitized"
        _sanitize(tmp / "model.pth", clean, daemon_python)
        os.replace(clean, tmp / "model.pth")
        for f in tmp.iterdir():
            os.chmod(f, 0o600)
        os.chmod(tmp, 0o700)
        if target.exists():
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
        return VoiceInfo(name=name, dir=target, manifest=manifest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    print(import_pack(Path(sys.argv[1])))
