import unittest
import struct
from opengnc.ground_segment.ccsds import SpacePacket, CUC, PacketType, SequenceFlags


class TestCCSDS(unittest.TestCase):
    def test_cuc_pack_unpack(self):
        t0 = 1711915200.5  # Some Unix timestamp
        packed = CUC.pack(t0)
        self.assertEqual(len(packed), 6)
        t1 = CUC.unpack(packed)
        self.assertAlmostEqual(t0, t1, places=4)

    def test_space_packet(self):
        data = b"\x01\x02\x03\x04"
        packet = SpacePacket(
            apid=0x123,
            packet_type=PacketType.TELEMETRY,
            sec_header_flag=True,
            seq_flags=SequenceFlags.UNSEGMENTED,
            seq_count=1000,
            data=data,
        )
        packed = packet.pack()
        self.assertEqual(len(packed), 6 + 4)

        # Manually check bits of first word (2 bytes)
        # Word1 = (0<<13) | (0<<12) | (1<<11) | 0x123 = 0x0800 | 0x0123 = 0x0923
        word1, word2, word3 = struct.unpack(">HHH", packed[:6])
        self.assertEqual(word1, 0x0923)
        self.assertEqual(word2, (3 << 14) | 1000)
        self.assertEqual(word3, len(data) - 1)

        unpacked = SpacePacket.unpack(packed)
        self.assertEqual(unpacked.apid, 0x123)
        self.assertEqual(unpacked.seq_count, 1000)
        self.assertEqual(unpacked.data, data)


if __name__ == "__main__":
    unittest.main()
