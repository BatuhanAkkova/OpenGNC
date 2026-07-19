"""
CCSDS Space Packet Protocol (Blue Book 133.0-B-2) and time code formats.
"""

from __future__ import annotations

import struct
import time
from enum import IntEnum


class PacketType(IntEnum):
    TELEMETRY = 0
    TELECOMMAND = 1


class SequenceFlags(IntEnum):
    CONTINUATION = 0
    FIRST = 1
    LAST = 2
    UNSEGMENTED = 3


class SpacePacket:
    HEADER_SIZE = 6

    def __init__(
        self,
        apid: int,
        packet_type: PacketType = PacketType.TELEMETRY,
        sec_header_flag: bool = False,
        seq_flags: SequenceFlags = SequenceFlags.UNSEGMENTED,
        seq_count: int = 0,
        data: bytes = b"",
    ) -> None:
        self.version = 0
        self.packet_type = packet_type
        self.sec_header_flag = sec_header_flag
        self.apid = apid & 0x07FF
        self.seq_flags = seq_flags
        self.seq_count = seq_count & 0x3FFF
        self.data = data

    @property
    def data_length_field(self) -> int:
        return max(0, len(self.data) - 1)

    def pack(self) -> bytes:
        word1 = (self.version << 13) | (self.packet_type << 12) | (int(self.sec_header_flag) << 11) | self.apid
        word2 = (self.seq_flags << 14) | self.seq_count
        word3 = self.data_length_field
        header = struct.pack(">HHH", word1, word2, word3)
        return header + self.data

    @classmethod
    def unpack(cls, buffer: bytes) -> SpacePacket:
        if len(buffer) < cls.HEADER_SIZE:
            raise ValueError("Buffer too short for CCSDS header.")

        word1, word2, word3 = struct.unpack(">HHH", buffer[:cls.HEADER_SIZE])

        version = (word1 >> 13) & 0x07
        packet_type = PacketType((word1 >> 12) & 0x01)
        sec_header_flag = bool((word1 >> 11) & 0x01)
        apid = word1 & 0x07FF

        seq_flags = SequenceFlags((word2 >> 14) & 0x03)
        seq_count = word2 & 0x3FFF

        data_len = word3 + 1
        data = buffer[cls.HEADER_SIZE : cls.HEADER_SIZE + data_len]

        packet = cls(apid, packet_type, sec_header_flag, seq_flags, seq_count, data)
        packet.version = version
        return packet


class CUC:
    SIZE = 6

    @staticmethod
    def pack(t: float | None = None) -> bytes:
        if t is None:
            t = time.time()

        coarse = int(t)
        fine = int((t - coarse) * 65536) & 0xFFFF
        return struct.pack(">IH", coarse, fine)

    @staticmethod
    def unpack(data: bytes) -> float:
        if len(data) < 6:
            raise ValueError("CUC data must be 6 bytes.")

        coarse, fine = struct.unpack(">IH", data[:6])
        return float(coarse) + (float(fine) / 65536.0)


__all__ = ["CUC", "PacketType", "SequenceFlags", "SpacePacket"]
