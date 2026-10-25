import pytest

from cyv import doctor
from cyv.cli import main as cli


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYV_HOME", str(tmp_path / "home"))


def test_doctor_runs_without_venv(capsys):
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "platform" in out and "ffmpeg" in out and "daemon venv" in out
    assert "fix: cyv setup" in out
    assert rc == 1  # no venv in a fresh CYV_HOME


def test_checks_structure():
    checks = {c.name: c for c in doctor.run_checks()}
    assert checks["daemon venv"].ok is False
    assert checks["Applio pin"].ok is False
    assert checks["MPS"].ok is None
    assert checks["python"].ok is True
