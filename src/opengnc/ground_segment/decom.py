"""
Telemetry De-commutation Engine.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


@dataclass
class TelemetryField:
    """Metadata for a single telemetry parameter."""
    name: str
    data_type: str  # 'f', 'd', 'H', 'h', 'B', 'b', 'I', 'i', 'Q', 'q'
    offset: int
    scale: float = 1.0
    offset_val: float = 0.0


class DecomEngine:
    """
    De-commutation Engine to extract parameters from a binary payload.
    """

    # Mapping of data types to their sizes in bytes
    TYPE_SIZES = {
        'f': 4, 'd': 8, 'H': 2, 'h': 2, 'B': 1, 'b': 1, 'I': 4, 'i': 4, 'Q': 8, 'q': 8
    }

    def __init__(self, fields: list[TelemetryField]) -> None:
        self.fields = fields

    def decommutate(self, payload: bytes) -> dict[str, Any]:
        """
        Extract parameters from the binary payload.

        Parameters
        ----------
        payload : bytes
            Raw binary data field of the CCSDS packet (excluding header).

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter names and their de-commutated values.
        """
        results = {}
        for field in self.fields:
            size = self.TYPE_SIZES.get(field.data_type, 0)
            if field.offset + size > len(payload):
                continue

            # Unpack using big-endian by default (common in space standards)
            fmt = f">{field.data_type}"
            raw_val = struct.unpack_from(fmt, payload, field.offset)[0]

            # Apply calibration (linear)
            results[field.name] = (raw_val * field.scale) + field.offset_val

        return results

    @classmethod
    def from_dict(cls, config: list[dict[str, Any]]) -> DecomEngine:
        """Create a DecomEngine from a list of configuration dictionaries."""
        fields = [TelemetryField(**f) for f in config]
        return cls(fields)
