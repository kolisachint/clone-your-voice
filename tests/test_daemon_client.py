"""Daemon + client integration in fake mode (no models)."""

from __future__ import annotations

import io
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from cyv import paths, protocol
from cyv.client import Client, DaemonError

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def daemon_env(cyv_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    monkeypatch.setenv("PYTHONPATH", SRC)

    def configure(idle_timeout: float = 60.0) -> None:
        monkeypatch.setenv(
            "CYV_DAEMON_CMD",
            f"{sys.executable} -m cyvd.server.main --fake --idle-timeout {idle_timeout}",
        )

    configure()
    yield configure
    try:
        pid = int(paths.pid_path().read_text())
        os.kill(pid, signal.SIGTERM)
        wait_for(lambda: not pid_alive(pid), 5)
    except (OSError, ValueError):
        pass


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:  # zombies count as dead
            return f.read().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def wait_for(cond: Callable[[], bool], timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def make_wav(n: int = 4800, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    x = (np.sin(np.arange(n) / 10) * 12000).astype("<i2")
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def wav_info(data: bytes) -> tuple[int, int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def test_health_and_spawn(daemon_env: Callable[..., None]) -> None:
    c = Client()
    assert c.status() == {"running": False}
    h = c.health()
    assert h["ok"] and h["device"] == "fake" and h["pid"] > 0
    assert c.status()["running"] is True


def test_say_returns_wav(daemon_env: Callable[..., None]) -> None:
    hdr, wav = Client().say("hello world", pitch=2)
    ch, width, sr, n = wav_info(wav)
    assert (ch, width) == (1, 2) and n > 0
    assert hdr["sample_rate"] == sr and hdr["audio_bytes"] == len(wav)
    _, longer = Client().say("hello world, this is a much longer sentence")
    assert wav_info(longer)[3] > n


def test_convert_roundtrip(daemon_env: Callable[..., None]) -> None:
    src = make_wav()
    _, out = Client().convert(src, watermark=False)
    assert wav_info(out) == wav_info(src)
    a = np.frombuffer(wave.open(io.BytesIO(src)).readframes(4800), "<i2")
    b = np.frombuffer(wave.open(io.BytesIO(out)).readframes(4800), "<i2")
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_bad_inputs(daemon_env: Callable[..., None]) -> None:
    c = Client()
    for voice in ("../etc", "UPPER", "a" * 33, ""):
        with pytest.raises(DaemonError) as ei:
            c.say("hi", voice=voice)
        assert ei.value.code == "bad_voice"
    with pytest.raises(DaemonError) as ei:
        c.convert(b"not a wav")
    assert ei.value.code == "bad_audio"
    with pytest.raises(DaemonError) as ei:
        c.say("   ")
    assert ei.value.code == "bad_request"
    assert c.health()["ok"]  # daemon survived


def test_oversize_frame_rejected_by_daemon(daemon_env: Callable[..., None]) -> None:
    Client().ensure_daemon()
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(paths.socket_path()))
    s.sendall((protocol.MAX_JSON_FRAME + 1).to_bytes(4, "big"))
    hdr = protocol.recv_json(s)
    s.close()
    assert hdr["ok"] is False and hdr["code"] == "bad_request"


def test_permissions_and_shutdown(daemon_env: Callable[..., None]) -> None:
    c = Client()
    c.ensure_daemon()
    assert stat.S_IMODE(paths.socket_path().stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.run_dir().stat().st_mode) == 0o700
    pid = int(paths.pid_path().read_text())
    assert c.shutdown() is True
    assert wait_for(lambda: not paths.socket_path().exists() and not pid_alive(pid), 5)
    assert not paths.pid_path().exists()
    assert c.shutdown() is False


def test_sigterm_cleans_up(daemon_env: Callable[..., None]) -> None:
    Client().ensure_daemon()
    pid = int(paths.pid_path().read_text())
    os.kill(pid, signal.SIGTERM)
    assert wait_for(lambda: not paths.socket_path().exists() and not pid_alive(pid), 5)


def test_concurrent_clients_spawn_one_daemon(daemon_env: Callable[..., None]) -> None:
    code = "from cyv.client import Client; print(Client().health()['pid'])"
    procs = [
        subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        for _ in range(6)
    ]
    pids = {p.communicate(timeout=30)[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs)
    assert len(pids) == 1
    assert pids == {paths.pid_path().read_text().strip()}


def test_idle_exit_then_respawn(daemon_env: Callable[..., None]) -> None:
    daemon_env(idle_timeout=0.5)
    c = Client()
    first = c.health()["pid"]
    assert wait_for(lambda: not pid_alive(first), 10)
    assert wait_for(lambda: not paths.socket_path().exists(), 2)
    second = c.health()["pid"]
    assert second != first


def test_inflight_request_survives_idle_deadline(
    daemon_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon_env(idle_timeout=0.3)
    monkeypatch.setenv("CYVD_FAKE_DELAY", "1.5")  # inherited by the spawned daemon
    result: dict[str, object] = {}

    def work() -> None:
        result["wav"] = Client().say("slow")[1]

    t = threading.Thread(target=work)
    t.start()
    t.join(30)
    assert wav_info(result["wav"])[3] > 0  # type: ignore[arg-type]


def test_stale_socket_recovered(daemon_env: Callable[..., None]) -> None:
    paths.ensure_private_dir(paths.run_dir())
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(paths.socket_path()))
    s.close()  # file remains, nobody listening
    assert paths.socket_path().exists()
    assert Client().health()["ok"]


def test_missing_daemon_command_is_readable(
    cyv_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CYV_DAEMON_CMD", "/nonexistent/python -m cyvd.server.main")
    with pytest.raises(DaemonError, match="cyv setup"):
        Client().health()
