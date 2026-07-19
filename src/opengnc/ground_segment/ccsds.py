"""
Deprecated compatibility wrapper for operational ground segment CCSDS helpers.
"""

from opengnc._compat import warn_legacy_import
from opengnc.ops.ground_segment.ccsds import CUC, PacketType, SequenceFlags, SpacePacket

warn_legacy_import("opengnc.ground_segment.ccsds", "opengnc.ops.ground_segment.ccsds")

__all__ = ["CUC", "PacketType", "SequenceFlags", "SpacePacket"]
