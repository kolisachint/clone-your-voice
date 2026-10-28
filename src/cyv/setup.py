"""`cyv setup`: build the daemon venv, fetch Applio at its pin, download base models."""

from __future__ import annotations

import hashlib
import http.client
import os
import platform
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import cyv
from cyv import ffmpeg, paths
from cyvd.timbre.applio.pin import APPLIO_COMMIT, APPLIO_REPO

PYTHON_VERSION = "3.12"


class SetupError(Exception):
    pass


@dataclass
class Step:
    title: str
    detail: str
    action: Callable[[], None]
    done: Callable[[], bool] = lambda: False


@dataclass
class Model:
    name: str
    url: str
    sha256: str
    dest: str
    size_bytes: int | None = None


CONSTRAINTS = Path(__file__).with_name("daemon-constraints.txt")


def is_mac_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def venv_python() -> Path:
    return paths.venv_dir() / "bin" / "python"


def applio_dir() -> Path:
    return paths.home() / "applio"


def find_uv() -> str | None:
    return shutil.which("uv")


def load_models() -> list[Model]:
    """Read models.toml; returns [] when it is absent. Accepts [[model]] or [name] tables."""
    try:
        text = resources.files("cyv").joinpath("models.toml").read_text("utf-8")
    except (FileNotFoundError, OSError):
        return []
    data = tomllib.loads(text)
    items: list[dict[str, Any]] = []
    for key in ("models", "model"):
        if isinstance(data.get(key), list):
            items = data[key]
        elif isinstance(data.get(key), dict):
            data = data[key]
    if not items:
        items = [{"name": k, **v} for k, v in data.items() if isinstance(v, dict)]
    models = []
    for it in items:
        try:
            size = it.get("size_bytes")
            models.append(
                Model(
                    it["name"],
                    it["url"],
                    it["sha256"].lower(),
                    it["dest"],
                    size if isinstance(size, int) and size > 0 else None,
                )
            )
        except KeyError as e:
            raise SetupError(f"models.toml entry {it.get('name', '?')!r} lacks {e}") from e
    return models


def model_path(m: Model) -> Path:
    root = paths.models_dir().resolve()
    p = (root / m.dest).resolve()
    if not p.is_relative_to(root):
        raise SetupError(f"model {m.name!r} dest escapes the models directory")
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def model_ok(m: Model) -> bool:
    p = model_path(m)
    return p.is_file() and sha256_file(p) == m.sha256


def _fetch_to_part(m: Model, part: Path) -> bool:
    """Stream into `part`, resuming if possible.

    Returns True when `part` now holds the full file, False when a stale/oversized partial
    was thrown away and the caller should start over.
    """
    have = part.stat().st_size if part.exists() else 0
    if m.size_bytes is not None and have > m.size_bytes:
        part.unlink(missing_ok=True)
        have = 0
    req = urllib.request.Request(  # noqa: S310
        m.url, headers={"User-Agent": f"cyv/{cyv.__version__}"}
    )
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (https enforced)
            resumed = have > 0 and getattr(resp, "status", 200) == 206
            h = hashlib.sha256()
            total = 0
            if resumed:
                h = hashlib.sha256(part.read_bytes())  # re-hash prefix; models are modest
                total = have
            with part.open("ab" if resumed else "wb") as f:
                while chunk := resp.read(1 << 20):
                    total += len(chunk)
                    if m.size_bytes is not None and total > m.size_bytes:
                        f.close()
                        part.unlink(missing_ok=True)
                        raise SetupError(
                            f"model {m.name!r}: download exceeded the expected "
                            f"{m.size_bytes} bytes; aborted"
                        )
                    f.write(chunk)
                    h.update(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            # Range not satisfiable: the .part is probably already complete.
            if sha256_file(part) == m.sha256:
                return True
            part.unlink(missing_ok=True)
            return False
        raise SetupError(f"model {m.name!r}: server returned HTTP {e.code} for {m.url}") from e
    except urllib.error.URLError as e:
        raise SetupError(
            f"model {m.name!r}: cannot download ({e.reason}); check your connection and re-run"
        ) from e
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        raise SetupError(
            f"model {m.name!r}: download interrupted ({e}); re-run `cyv setup` to resume"
        ) from e
    return True


def download_model(m: Model, *, progress: Callable[[str], None] = print) -> None:
    """Stream to <dest>.part (resuming if possible), verify sha256, rename atomically."""
    if not m.url.startswith("https://"):
        raise SetupError(f"model {m.name!r}: only https URLs are allowed")
    dest = model_path(m)
    if model_ok(m):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if not _fetch_to_part(m, part):  # stale partial discarded: restart once from zero
        if not _fetch_to_part(m, part):
            raise SetupError(f"model {m.name!r}: could not complete the download")
    if sha256_file(part) != m.sha256:
        part.unlink(missing_ok=True)
        raise SetupError(f"sha256 mismatch for {m.name}; deleted partial download, retry setup")
    os.replace(part, dest)
    progress(f"  downloaded {m.name}")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SetupError(f"command failed: {' '.join(cmd)} ({e})") from e


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=applio_dir(), capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def applio_ok() -> bool:
    return (applio_dir() / ".git").exists() and _git("rev-parse", "HEAD") == APPLIO_COMMIT


def project_spec() -> str:
    """Install from the source checkout if running from one, else from PyPI."""
    root = Path(cyv.__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "cyvd").is_dir():
        return f"{root}[daemon]"
    return "clone-your-voice[daemon]"


def _install_applio() -> None:
    d = applio_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".git").exists():
        _run(["git", "init", "-q"], cwd=d)
    remotes = _git("remote")
    if "origin" not in remotes.split():
        _run(["git", "remote", "add", "origin", APPLIO_REPO], cwd=d)
    _run(["git", "fetch", "--depth", "1", "origin", APPLIO_COMMIT], cwd=d)
    _run(["git", "checkout", "-q", "--force", APPLIO_COMMIT], cwd=d)


def _check_platform(force: bool) -> None:
    if is_mac_arm64():
        return
    here = f"{sys.platform}/{platform.machine()}"
    if not force:
        raise SetupError(
            f"cyv v1 supports macOS on Apple Silicon only (this is {here}). "
            "Re-run with --force to try anyway (unsupported)."
        )
    print(f"warning: unsupported platform {here}; continuing because of --force", file=sys.stderr)


def _check_xcode() -> None:
    try:
        r = subprocess.run(["xcode-select", "-p"], capture_output=True, check=False)
    except OSError:
        r = None
    if r is None or r.returncode != 0:
        raise SetupError("Xcode Command Line Tools missing; run: xcode-select --install")


def _check_ffmpeg() -> None:
    try:
        ffmpeg.find_ffmpeg()
    except ffmpeg.FfmpegError as e:
        raise SetupError(str(e)) from e


def _check_uv() -> None:
    if find_uv() is None:
        raise SetupError(
            "uv not found; install it with: brew install uv "
            "(or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
        )


def plan(*, force: bool = False) -> list[Step]:
    uv = find_uv() or "uv"
    vpy = venv_python()
    steps = [
        Step(
            "Check platform",
            "macOS on Apple Silicon (arm64)",
            lambda: _check_platform(force),
        ),
        Step("Check ffmpeg", "ffmpeg on PATH (brew install ffmpeg)", _check_ffmpeg),
    ]
    if sys.platform == "darwin":
        steps.append(Step("Check Xcode CLT", "xcode-select -p", _check_xcode))
    steps += [
        Step("Find uv", "uv on PATH (brew install uv)", _check_uv),
        Step(
            "Create daemon venv",
            f"{uv} venv --python {PYTHON_VERSION} {paths.venv_dir()}",
            lambda: _run([uv, "venv", "--python", PYTHON_VERSION, str(paths.venv_dir())]),
            vpy.exists,
        ),
        Step(
            "Install cyv + daemon dependencies",
            f"{uv} pip install --python {vpy} -c {CONSTRAINTS} '{project_spec()}'",
            lambda: _run(
                [uv, "pip", "install", "--python", str(vpy), "-c", str(CONSTRAINTS), project_spec()]
            ),
        ),
        Step(
            f"Fetch Applio @ {APPLIO_COMMIT[:10]}",
            f"git init + fetch --depth 1 + checkout into {applio_dir()}",
            _install_applio,
            applio_ok,
        ),
        Step(
            "Install Applio engine requirements",
            f"{uv} pip install --python {vpy} -c {CONSTRAINTS} "
            f"-r {applio_dir()}/requirements-engine.txt",
            lambda: _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(vpy),
                    "-c",
                    str(CONSTRAINTS),
                    "-r",
                    str(applio_dir() / "requirements-engine.txt"),
                ]
            ),  # fmt: skip
        ),
    ]
    for m in load_models():
        steps.append(
            Step(
                f"Download model {m.name}",
                f"{m.url} -> {paths.models_dir() / m.dest} (sha256 {m.sha256[:12]}...)",
                lambda m=m: download_model(m),
                lambda m=m: model_ok(m),
            )
        )
    return steps


def run_setup(*, dry_run: bool = False, force: bool = False) -> int:
    if not is_mac_arm64() and not force:
        _check_platform(force)  # raises with guidance
    steps = plan(force=force)
    total = len(steps)
    for i, s in enumerate(steps, 1):
        if dry_run:
            print(f"[{i}/{total}] {s.title}\n      {s.detail}")
            continue
        print(f"[{i}/{total}] {s.title}")
        if s.done():
            print("  already done")
            continue
        s.action()
    if dry_run:
        print("\n(dry run: nothing was changed)")
    else:
        print("\nSetup complete. Try: cyv doctor")
    return 0
