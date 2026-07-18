"""
CCSDS Space Packet Protocol (Blue Book 133.0-B-2) and Time Code Formats.
"""

from __future__ import annotations

import struct
import time
from enum import IntEnum


class PacketType(IntEnum):
    """CCSDS Packet Type."""
    TELEMETRY = 0
    TELECOMMAND = 1


class SequenceFlags(IntEnum):
    """CCSDS Sequence Flags."""
    CONTINUATION = 0
    FIRST = 1
    LAST = 2
    UNSEGMENTED = 3


class SpacePacket:
    """
    Implementation of the CCSDS Space Packet Primary Header.

    Total header size: 6 octets (48 bits).
    """

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
        """The value in the length field is (Total Data Field Octets - 1)."""
        return max(0, len(self.data) - 1)

    def pack(self) -> bytes:
        """Pack the header and data into a binary buffer."""
        # Octet 0-1: Version (3), Type (1), Sec Header (1), APID (11)
        word1 = (self.version << 13) | (self.packet_type << 12) | (int(self.sec_header_flag) << 11) | self.apid
        # Octet 2-3: Seq Flags (2), Seq Count (14)
        word2 = (self.seq_flags << 14) | self.seq_count
        # Octet 4-5: Length (16)
        word3 = self.data_length_field

        header = struct.pack(">HHH", word1, word2, word3)
        return header + self.data

    @classmethod
    def unpack(cls, buffer: bytes) -> SpacePacket:
        """Unpack a binary buffer into a SpacePacket instance."""
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
    """
    CCSDS Unsegmented Time Code (CUC).

    Format: 4 octets coarse time (seconds), 2 octets fine time (1/65536 subseconds).
    Epoch: TAI (1958 Jan 1) or Unix (1970 Jan 1). Here we use Unix for convenience.
    """

    SIZE = 6

    @staticmethod
    def pack(t: float | None = None) -> bytes:
        """Pack a float timestamp into 6-byte CUC."""
        if t is None:
            t = time.time()

        coarse = int(t)
        fine = int((t - coarse) * 65536) & 0xFFFF
        return struct.pack(">IH", coarse, fine)

    @staticmethod
    def unpack(data: bytes) -> float:
        """Unpack 6-byte CUC into a float timestamp."""
        if len(data) < 6:
            raise ValueError("CUC data must be 6 bytes.")

        coarse, fine = struct.unpack(">IH", data[:6])
        return float(coarse) + (float(fine) / 65536.0)
