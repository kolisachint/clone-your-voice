from __future__ import annotations

import asyncio
import socket

import pytest

from cyv import protocol


def test_frame_roundtrip_blocking() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.sendall(protocol.encode_json({"op": "health", "n": 1}))
        a.sendall(protocol.encode_frame(b"\x00\x01binary"))
        assert protocol.recv_json(b) == {"op": "health", "n": 1}
        assert protocol.recv_audio(b) == b"\x00\x01binary"


def test_frame_roundtrip_asyncio() -> None:
    async def run() -> None:
        r = asyncio.StreamReader()
        r.feed_data(protocol.encode_json({"ok": True}) + protocol.encode_frame(b"abc"))
        assert await protocol.read_json(r) == {"ok": True}
        assert await protocol.read_audio(r) == b"abc"

    asyncio.run(run())


def test_oversize_json_rejected() -> None:
    async def run() -> None:
        r = asyncio.StreamReader()
        r.feed_data((protocol.MAX_JSON_FRAME + 1).to_bytes(4, "big"))
        with pytest.raises(protocol.ProtocolError):
            await protocol.read_json(r)

    asyncio.run(run())


def test_oversize_blocking_rejected() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.sendall((protocol.MAX_AUDIO_FRAME + 1).to_bytes(4, "big"))
        with pytest.raises(protocol.ProtocolError):
            protocol.recv_audio(b)


def test_non_object_json_rejected() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.sendall(protocol.encode_frame(b"[1,2]"))
        with pytest.raises(protocol.ProtocolError):
            protocol.recv_json(b)


def test_truncated_frame_is_reset() -> None:
    a, b = socket.socketpair()
    with b:
        a.sendall((10).to_bytes(4, "big") + b"abc")
        a.close()
        with pytest.raises(ConnectionResetError):
            protocol.recv_json(b)
