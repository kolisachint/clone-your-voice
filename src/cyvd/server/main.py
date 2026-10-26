"""cyvd entry point: `python -m cyvd.server.main [--keep-alive] [--idle-timeout S] [--fake]`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
import struct
import sys
import time
from typing import Any

from cyv import paths, protocol
from cyvd import __version__
from cyvd.server.peercred import peer_uid
from cyvd.server.pipeline import Pipeline, RequestError, build_pipeline, validate_voice_name
from cyvd.server.wavio import WavError, wav_sample_rate

log = logging.getLogger("cyvd")

DEFAULT_IDLE_TIMEOUT = 900.0
MAX_TEXT_CHARS = 20_000
HEADER_TIMEOUT = 30.0  # seconds to receive a JSON header
AUDIO_BASE_TIMEOUT = 30.0  # audio frame: base + size / AUDIO_MIN_RATE
AUDIO_MIN_RATE = 1 << 20  # bytes/s a client must sustain
MAX_BUFFERED_AUDIO = 2  # concurrent convert requests holding an audio frame in memory


class Daemon:
    def __init__(self, pipeline: Pipeline, idle_timeout: float, keep_alive: bool) -> None:
        self.pipeline = pipeline
        self.idle_timeout = idle_timeout
        self.keep_alive = keep_alive
        self.started = time.monotonic()
        self.last_activity = time.monotonic()
        self.active = 0  # requests doing actual work (queued for or running on the worker)
        self.audio_sem = asyncio.Semaphore(MAX_BUFFERED_AUDIO)
        self.work_lock = asyncio.Lock()  # serializes GPU work
        self.stop = asyncio.Event()
        self.uid = os.getuid()

    # --- connection handling -------------------------------------------------

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        shutdown = False
        try:
            sock = writer.get_extra_info("socket")
            if sock is None or peer_uid(sock) != self.uid:
                log.warning("rejected connection from foreign uid")
                return
            try:
                req = await asyncio.wait_for(protocol.read_json(reader), HEADER_TIMEOUT)
                header, audio = await self.dispatch(req, reader)
                shutdown = req.get("op") == "shutdown"
            except RequestError as e:
                header, audio = {"ok": False, "error": str(e), "code": e.code}, b""
            except TimeoutError:
                header, audio = {"ok": False, "error": "request timed out", "code": "timeout"}, b""
            except protocol.ProtocolError as e:
                header, audio = {"ok": False, "error": str(e), "code": "bad_request"}, b""
            except WavError as e:
                header, audio = {"ok": False, "error": str(e), "code": "bad_audio"}, b""
            except Exception as e:  # report, never crash the daemon
                log.exception("request failed")
                header = {"ok": False, "error": f"{type(e).__name__}: {e}", "code": "internal"}
                audio = b""
            if audio:
                header["audio_bytes"] = len(audio)
                header["sample_rate"] = wav_sample_rate(audio)
            writer.write(protocol.encode_json(header))
            if audio:
                writer.write(protocol.encode_frame(audio))
            await asyncio.wait_for(writer.drain(), AUDIO_BASE_TIMEOUT + len(audio) / AUDIO_MIN_RATE)
        except (asyncio.IncompleteReadError, ConnectionError, TimeoutError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
            self.last_activity = time.monotonic()
            if shutdown:
                self.stop.set()

    async def run_work(self, fn: Any, *args: Any) -> Any:
        self.active += 1  # only real work counts; idle or stalled sockets never block idle-exit
        try:
            async with self.work_lock:
                return await asyncio.get_running_loop().run_in_executor(None, fn, *args)
        finally:
            self.active -= 1
            self.last_activity = time.monotonic()

    @staticmethod
    async def read_audio_frame(reader: asyncio.StreamReader) -> bytes:
        """Read one audio frame with a size-based timeout (the size is known after 4 bytes)."""
        (n,) = struct.Struct(">I").unpack(
            await asyncio.wait_for(reader.readexactly(4), HEADER_TIMEOUT)
        )
        if n > protocol.MAX_AUDIO_FRAME:
            raise protocol.ProtocolError(f"frame too large: {n} > {protocol.MAX_AUDIO_FRAME}")
        return await asyncio.wait_for(
            reader.readexactly(n), AUDIO_BASE_TIMEOUT + n / AUDIO_MIN_RATE
        )

    async def dispatch(
        self, req: dict[str, Any], reader: asyncio.StreamReader
    ) -> tuple[dict[str, Any], bytes]:
        op = req.get("op")
        if op == "health":
            return {
                "ok": True,
                "version": __version__,
                "protocol": protocol.PROTOCOL_VERSION,
                "device": self.pipeline.device,
                "voice": self.pipeline.loaded,
                "uptime_s": round(time.monotonic() - self.started, 1),
                "pid": os.getpid(),
            }, b""
        if op == "shutdown":
            return {"ok": True}, b""
        if op == "reload":
            await self.run_work(self.pipeline.reload, _voice(req))
            return {"ok": True}, b""
        if op == "say":
            voice, pitch, wm = _voice(req), _pitch(req), _watermark(req)
            text = req.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RequestError("bad_request", "text must be a non-empty string")
            if len(text) > MAX_TEXT_CHARS:
                raise RequestError("bad_request", f"text longer than {MAX_TEXT_CHARS} chars")
            return {"ok": True}, await self.run_work(self.pipeline.say, text, voice, pitch, wm)
        if op == "convert":
            n = req.get("audio_bytes")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise RequestError("bad_request", "convert needs integer audio_bytes >= 0")
            async with self.audio_sem:
                # Always consume the declared frame before replying, even if the request is
                # about to be rejected: otherwise the client's send fails and it never sees
                # the error. The read is bounded by MAX_AUDIO_FRAME and the timeouts above.
                data = await self.read_audio_frame(reader)
                voice, pitch, wm = _voice(req), _pitch(req), _watermark(req)
                if not data:
                    raise WavError("audio is empty")
                if len(data) != n:
                    raise RequestError("bad_request", "audio_bytes does not match audio frame")
                wav = await self.run_work(self.pipeline.convert, data, voice, pitch, wm)
            return {"ok": True}, wav
        if op == "detect":
            n = req.get("audio_bytes")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise RequestError("bad_request", "detect needs integer audio_bytes >= 0")
            async with self.audio_sem:
                # Same framing discipline as convert: consume the frame before replying.
                data = await self.read_audio_frame(reader)
                if not data:
                    raise WavError("audio is empty")
                if len(data) != n:
                    raise RequestError("bad_request", "audio_bytes does not match audio frame")
                result = await self.run_work(self.pipeline.detect, data)
            return {"ok": True, **result}, b""
        raise RequestError("bad_request", f"unknown op: {op!r}")

    # --- lifecycle -----------------------------------------------------------

    async def idle_watch(self) -> None:
        tick = max(0.02, min(1.0, self.idle_timeout / 4))
        while not self.stop.is_set():
            await asyncio.sleep(tick)
            idle = (
                self.active == 0
                and not self.work_lock.locked()
                and time.monotonic() - self.last_activity >= self.idle_timeout
            )
            if idle:
                log.info("idle for %.0fs, exiting", self.idle_timeout)
                self.stop.set()


def _voice(req: dict[str, Any]) -> str | None:
    v = req.get("voice")
    return None if v is None else validate_voice_name(v)


def _pitch(req: dict[str, Any]) -> int:
    p = req.get("pitch", 0)
    if isinstance(p, bool) or not isinstance(p, int) or not -48 <= p <= 48:
        raise RequestError("bad_request", "pitch must be an integer in [-48, 48]")
    return p


def _watermark(req: dict[str, Any]) -> bool:
    w = req.get("watermark", True)
    if not isinstance(w, bool):
        raise RequestError("bad_request", "watermark must be a boolean")
    return w


def _socket_is_live(path: str) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(path)
    except OSError:
        return False
    finally:
        s.close()
    return True


async def serve(args: argparse.Namespace) -> int:
    run = paths.ensure_private_dir(paths.run_dir())
    sock_path = paths.socket_path()
    pid_path = paths.pid_path()
    pipeline = build_pipeline(args.fake)

    if sock_path.exists() or sock_path.is_symlink():
        if _socket_is_live(str(sock_path)):
            log.error("another daemon is already listening on %s", sock_path)
            return 1
        sock_path.unlink()

    d = Daemon(pipeline, args.idle_timeout, args.keep_alive)
    old_umask = os.umask(0o177)  # socket is created 0600
    try:
        server = await asyncio.start_unix_server(d.handle, path=str(sock_path))
    finally:
        os.umask(old_umask)
    os.chmod(sock_path, 0o600)
    sock_ino = sock_path.stat().st_ino
    pid_path.write_text(f"{os.getpid()}\n")
    os.chmod(pid_path, 0o600)
    log.info("listening on %s (run dir %s)", sock_path, run)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, d.stop.set)
    watcher = None if args.keep_alive else asyncio.create_task(d.idle_watch())

    try:
        await d.stop.wait()
    finally:
        server.close()
        deadline = time.monotonic() + 5
        while d.active and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if watcher:
            watcher.cancel()
        # Only remove files that are still ours (a successor may have replaced them).
        with contextlib.suppress(OSError):
            if sock_path.stat().st_ino == sock_ino:
                sock_path.unlink()
        with contextlib.suppress(OSError):
            if pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cyvd")
    p.add_argument("--keep-alive", action="store_true", help="never exit on idle")
    p.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT, metavar="S")
    p.add_argument("--fake", action="store_true", help="use model-free fakes (tests)")
    args = p.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)s cyvd %(levelname)s %(message)s"
    )
    sys.exit(asyncio.run(serve(args)))


if __name__ == "__main__":
    main()
