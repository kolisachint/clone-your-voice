from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def cyv_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short temp CYV_HOME (unix socket paths have a ~104 byte limit)."""
    d = Path(tempfile.mkdtemp(prefix="cyv", dir="/tmp"))
    monkeypatch.setenv("CYV_HOME", str(d))
    yield d
    shutil.rmtree(d, ignore_errors=True)
