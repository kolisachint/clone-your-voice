"""Peer uid lookup for unix sockets (Linux SO_PEERCRED, macOS LOCAL_PEERCRED)."""

from __future__ import annotations

import socket
import struct
import sys

_LOCAL_PEERCRED = 0x001  # <sys/un.h>
_SOL_LOCAL = 0  # level for LOCAL_* options on macOS
_XUCRED_SIZE = 76  # struct xucred: u32 version, u32 uid, i16 ngroups, u32 groups[16]


def peer_uid(sock: socket.socket) -> int:
    if sys.platform == "darwin":
        raw = sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERCRED, _XUCRED_SIZE)
        _version, uid = struct.unpack_from("=II", raw)
        return uid
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", raw)
    return uid
