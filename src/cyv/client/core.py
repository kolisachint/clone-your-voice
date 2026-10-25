"""Blocking client: connect, spawn-on-demand under flock, one request per connection."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import shlex
import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any

from cyv import paths, protocol

_NOT_RUNNING = (errno.ECONNREFUSED, errno.ENOENT)
_RETRYABLE = (ConnectionResetError, BrokenPipeError, ConnectionRefusedError, FileNotFoundError)


class DaemonError(Exception):
    """Readable failure talking to (or starting) the daemon."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def daemon_command() -> list[str]:
    env = os.environ.get("CYV_DAEMON_CMD")
    if env:
        return shlex.split(env)
    return [str(paths.venv_dir() / "bin" / "python"), "-m", "cyvd.server.main"]


class Client:
    def __init__(self, request_timeout: float = 600.0) -> None:
        self.request_timeout = request_timeout

    # --- public API ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request({"op": "health"})[0]

    def status(self) -> dict[str, Any]:
        """Health if the daemon is running; never spawns."""
        try:
            return {"running": True, **self._exchange({"op": "health"}, None, timeout=5)[0]}
        except OSError:
            return {"running": False}

    def say(
        self, text: str, voice: str | None = None, pitch: int = 0, watermark: bool = True
    ) -> tuple[dict[str, Any], bytes]:
        req = {"op": "say", "text": text, "voice": voice, "pitch": pitch, "watermark": watermark}
        return self._request(req)

    def convert(
        self, wav_bytes: bytes, voice: str | None = None, pitch: int = 0, watermark: bool = True
    ) -> tuple[dict[str, Any], bytes]:
        req = {
            "op": "convert",
            "audio_bytes": len(wav_bytes),
            "voice": voice,
            "pitch": pitch,
            "watermark": watermark,
        }
        return self._request(req, wav_bytes)

    def reload(self, voice: str | None = None) -> None:
        self._request({"op": "reload", "voice": voice})

    def shutdown(self) -> bool:
        """Ask a running daemon to exit. Returns False if none was running."""
        try:
            self._exchange({"op": "shutdown"}, None, timeout=10)
        except OSError:
            return False
        return True

    def ensure_daemon(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        paths.ensure_private_dir(paths.run_dir())
        with self._locked(deadline):
            proc: subprocess.Popen[bytes] | None = None
            while True:
                try:
                    self._exchange({"op": "health"}, None, timeout=5)
                    return
                except OSError as e:
                    refused = e.errno in _NOT_RUNNING
                    if refused and proc is None:
                        proc = self._spawn()
                    elif proc is not None and proc.poll() is not None:
                        raise DaemonError(
                            f"daemon exited during startup (code {proc.returncode}); "
                            f"see {paths.run_dir() / 'cyvd.log'}"
                        ) from e
                except protocol.ProtocolError:
                    pass
                if time.monotonic() > deadline:
                    raise DaemonError(
                        f"daemon did not become ready in {timeout:.0f}s; "
                        f"see {paths.run_dir() / 'cyvd.log'}"
                    )
                time.sleep(0.05)

    # --- internals -------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self, deadline: float) -> Iterator[None]:
        fd = os.open(paths.lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise DaemonError(
                            "timed out waiting for another cyv to start the daemon"
                        ) from None
                    time.sleep(0.05)
            yield
        finally:
            os.close(fd)  # releases the flock

    def _spawn(self) -> subprocess.Popen[bytes]:
        with contextlib.suppress(FileNotFoundError):
            paths.socket_path().unlink()  # stale socket from a dead daemon
        cmd = daemon_command()
        log = os.open(paths.run_dir() / "cyvd.log", os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as e:
            raise DaemonError(
                f"cannot start daemon ({cmd[0]}): {e.strerror}. Run `cyv setup` first."
            ) from e
        finally:
            os.close(log)

    def _exchange(
        self, req: dict[str, Any], audio: bytes | None, timeout: float | None = None
    ) -> tuple[dict[str, Any], bytes]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout or self.request_timeout)
        try:
            s.connect(str(paths.socket_path()))
            s.sendall(protocol.encode_json(req))
            if audio is not None:
                s.sendall(protocol.encode_frame(audio))
            header = protocol.recv_json(s)
            body = b""
            if header.get("ok") and header.get("audio_bytes", 0) > 0:
                body = protocol.recv_audio(s)
            return header, body
        finally:
            s.close()

    def _request(
        self, req: dict[str, Any], audio: bytes | None = None
    ) -> tuple[dict[str, Any], bytes]:
        try:
            try:
                header, body = self._exchange(req, audio)
            except _RETRYABLE:
                # not running, or the daemon idle-exited under us: (re)start and retry once
                self.ensure_daemon()
                header, body = self._exchange(req, audio)
        except TimeoutError as e:
            raise DaemonError("daemon request timed out") from e
        except (OSError, protocol.ProtocolError) as e:
            raise DaemonError(f"cannot talk to daemon: {e}") from e
        if not header.get("ok"):
            raise DaemonError(header.get("error", "unknown daemon error"), header.get("code"))
        return header, body
