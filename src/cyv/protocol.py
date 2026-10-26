"""Wire protocol between `cyv` (client) and `cyvd` (daemon). See ADR-003.

Every message is a sequence of frames. A frame is a 4-byte big-endian unsigned
length followed by that many bytes.

Request:  one JSON frame  {"op": <str>, ...args}
Response: one JSON frame  {"ok": true, ...result}  or  {"ok": false, "error": <str>, "code": <str>}
          If the JSON header contains "audio_bytes": N (N > 0), exactly one binary frame
          of N bytes follows, holding a mono 16-bit PCM WAV file at header["sample_rate"].

Ops (v1):
  health   -> {"ok", "version", "device", "voice": <name|null>, "uptime_s"}
  say      {text, voice, pitch=0, watermark=true}           -> header + WAV
  convert  {audio_bytes: N, voice, pitch=0, watermark=true} + one binary frame (input WAV)
                                                             -> header + WAV
  detect   {audio_bytes: N} + one binary frame (input WAV)
                                                             -> {"ok", "prob", "message", "crc_ok", "backend"}
                                                             (JSON only: prob is the watermarked fraction,
                                                             message the decoded 16-bit int or null, crc_ok
                                                             whether it is a coherent cyv message)
  reload   {voice?}                                          -> {"ok"}
  shutdown                                                   -> {"ok"}

The daemon never receives or writes output filesystem paths; the client writes files.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import Any

PROTOCOL_VERSION = 1
MAX_JSON_FRAME = 1 << 20  # 1 MiB
MAX_AUDIO_FRAME = 512 << 20  # 512 MiB
_HDR = struct.Struct(">I")


class ProtocolError(Exception):
    pass


def encode_frame(payload: bytes) -> bytes:
    return _HDR.pack(len(payload)) + payload


def encode_json(obj: dict[str, Any]) -> bytes:
    return encode_frame(json.dumps(obj, separators=(",", ":")).encode())


def _decode_json(raw: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad json frame: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("json frame must be an object")
    return obj


# --- asyncio (daemon side) ---------------------------------------------------


async def read_frame(reader: asyncio.StreamReader, limit: int) -> bytes:
    (n,) = _HDR.unpack(await reader.readexactly(_HDR.size))
    if n > limit:
        raise ProtocolError(f"frame too large: {n} > {limit}")
    return await reader.readexactly(n)


async def read_json(reader: asyncio.StreamReader) -> dict[str, Any]:
    return _decode_json(await read_frame(reader, MAX_JSON_FRAME))


async def read_audio(reader: asyncio.StreamReader) -> bytes:
    return await read_frame(reader, MAX_AUDIO_FRAME)


# --- blocking sockets (client side) -----------------------------------------


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 16))
        if not chunk:
            raise ConnectionResetError("daemon closed connection")
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket, limit: int) -> bytes:
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if n > limit:
        raise ProtocolError(f"frame too large: {n} > {limit}")
    return _recv_exact(sock, n)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    return _decode_json(recv_frame(sock, MAX_JSON_FRAME))


def recv_audio(sock: socket.socket) -> bytes:
    return recv_frame(sock, MAX_AUDIO_FRAME)
