"""Filesystem layout under CYV_HOME (default ~/.cyv)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# macOS sun_path is 104 bytes incl. NUL; Linux is 108. Stay under the smaller.
_MAX_SOCK_PATH = 100


def home() -> Path:
    """CYV_HOME as an absolute, resolved path (the engine chdirs, so never relative)."""
    return Path(os.environ.get("CYV_HOME") or Path.home() / ".cyv").expanduser().resolve()


def voices_dir() -> Path:
    return home() / "voices"


def models_dir() -> Path:
    """Shared base models (HuBERT/ContentVec, RMVPE, AudioSeal)."""
    return home() / "models"


def venv_dir() -> Path:
    """Daemon venv built by `cyv setup`."""
    return home() / "venv"


def config_file() -> Path:
    return home() / "config.toml"


def run_dir() -> Path:
    """Directory for socket, lock and pid. Created 0700 before use."""
    d = home() / "run"
    if len(str(d / "cyvd.sock")) > _MAX_SOCK_PATH:
        d = Path(tempfile.gettempdir()) / f"cyv-{os.getuid()}"
    return d


def ensure_private_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def socket_path() -> Path:
    return run_dir() / "cyvd.sock"


def lock_path() -> Path:
    return run_dir() / "cyvd.lock"


def pid_path() -> Path:
    return run_dir() / "cyvd.pid"
