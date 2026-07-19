"""
Deprecated compatibility exports for operational ground segment modules.
"""

from opengnc._compat import warn_legacy_import
from opengnc.ops.ground_segment import (
    CUC,
    DecomEngine,
    PacketType,
    SequenceFlags,
    SpacePacket,
    TelemetryField,
)

warn_legacy_import("opengnc.ground_segment", "opengnc.ops.ground_segment")

__all__ = [
    "CUC",
    "DecomEngine",
    "PacketType",
    "SequenceFlags",
    "SpacePacket",
    "TelemetryField",
]
