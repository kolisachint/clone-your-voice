"""The 16-bit cyv watermark payload.

Bit layout (MSB first, bit 15 is the leftmost bit)::

    15..12  format version   (currently 1)
    11..8   flags            (bit 8 = synthetic voice, always set by cyv; bits 9..11 reserved = 0)
     7..0   CRC-8 (poly 0x07, init 0) of the high byte (version << 4 | flags)

The payload is constant per cyv format version. It never carries anything about the
user, the voice, the text or the machine. The CRC lets a detector tell a coherent cyv
message from bit noise.
"""

from __future__ import annotations

FORMAT_VERSION = 1
FLAG_SYNTHETIC = 0b0001
NBITS = 16


def _crc8(byte: int) -> int:
    crc = byte & 0xFF
    for _ in range(8):
        crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def pack(version: int = FORMAT_VERSION, flags: int = FLAG_SYNTHETIC) -> int:
    if not 0 <= version < 16 or not 0 <= flags < 16:
        raise ValueError("version and flags are 4-bit fields")
    high = (version << 4) | flags
    return (high << 8) | _crc8(high)


def unpack(message: int) -> tuple[int, int, bool]:
    """Return (version, flags, crc_ok)."""
    high, crc = (message >> 8) & 0xFF, message & 0xFF
    return high >> 4, high & 0xF, crc == _crc8(high)


CYV_MESSAGE: int = pack()
