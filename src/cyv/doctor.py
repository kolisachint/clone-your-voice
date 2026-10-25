"""`cyv doctor`: diagnose the install and print the exact fix for each failing check."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass

from cyv import ffmpeg, paths, setup


@dataclass
class Check:
    name: str
    ok: bool | None  # None = informational / not applicable
    detail: str
    fix: str = ""


def _platform() -> list[Check]:
    arch = platform.machine()
    ok = setup.is_mac_arm64()
    checks = [
        Check(
            "platform",
            ok,
            f"{sys.platform} {arch}",
            "" if ok else "cyv v1 requires macOS on Apple Silicon",
        )
    ]
    if sys.platform == "darwin":
        checks.append(Check("macOS version", True, platform.mac_ver()[0] or "unknown"))
    checks.append(Check("python", True, platform.python_version()))
    return checks


def _tools() -> list[Check]:
    out = []
    v = ffmpeg.version()
    out.append(Check("ffmpeg", v is not None, v or "not found", "brew install ffmpeg"))
    uv = setup.find_uv()
    out.append(Check("uv", uv is not None, uv or "not found", "brew install uv"))
    return out


def _venv() -> list[Check]:
    vpy = setup.venv_python()
    ok = vpy.exists()
    checks = [Check("daemon venv", ok, str(paths.venv_dir()) if ok else "missing", "cyv setup")]
    ad = setup.applio_dir()
    if ad.exists():
        head = setup._git("rev-parse", "HEAD")
        match = head == setup.APPLIO_COMMIT
        detail = f"{head[:10] or 'no checkout'} (pin {setup.APPLIO_COMMIT[:10]})"
        checks.append(Check("Applio pin", match, detail, "" if match else "cyv setup"))
    else:
        checks.append(Check("Applio pin", False, "not present", "cyv setup"))
    try:
        models = setup.load_models()
    except setup.SetupError as e:
        models = []
        checks.append(Check("models", False, str(e)))
    bad = [m.name for m in models if not setup.model_ok(m)]
    if models:
        detail = "all present and verified" if not bad else "missing/corrupt: " + ", ".join(bad)
        checks.append(Check("models", not bad, detail, "" if not bad else "cyv setup"))
    else:
        checks.append(Check("models", None, "no model list available"))
    return checks


def _mps() -> Check:
    vpy = setup.venv_python()
    if not vpy.exists():
        return Check("MPS", None, "skipped (no daemon venv)")
    code = "import torch; print(torch.backends.mps.is_available())"
    try:
        r = subprocess.run([str(vpy), "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return Check("MPS", False, f"could not run venv python: {e}", "cyv setup")
    if r.returncode != 0:
        return Check("MPS", False, "torch failed to import in the venv", "cyv setup")
    avail = r.stdout.strip() == "True"
    return Check(
        "MPS",
        avail if sys.platform == "darwin" else None,
        "available" if avail else "not available (CPU only)",
        "update macOS / check `cyv setup` log" if sys.platform == "darwin" and not avail else "",
    )


def _daemon() -> Check:
    try:
        from cyv.client import Client

        status = Client().status()
    except Exception:
        return Check("daemon", None, "not running (starts on demand)", "cyv serve")
    return Check("daemon", True, f"running {status}" if status else "running")


def run_checks() -> list[Check]:
    return [*_platform(), *_tools(), *_venv(), _mps(), _daemon()]


def run_doctor() -> int:
    checks = run_checks()
    for c in checks:
        mark = {True: "ok  ", False: "FAIL", None: "info"}[c.ok]
        print(f"[{mark}] {c.name}: {c.detail}")
        if c.ok is False and c.fix:
            print(f"       fix: {c.fix}")
    failed = [c for c in checks if c.ok is False]
    print(f"\n{len(failed)} problem(s) found." if failed else "\nAll checks passed.")
    return 1 if failed else 0
