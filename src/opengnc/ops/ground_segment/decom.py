"""
Telemetry de-commutation engine.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


@dataclass
class TelemetryField:
    name: str
    data_type: str
    offset: int
    scale: float = 1.0
    offset_val: float = 0.0


class DecomEngine:
    TYPE_SIZES = {
        "f": 4,
        "d": 8,
        "H": 2,
        "h": 2,
        "B": 1,
        "b": 1,
        "I": 4,
        "i": 4,
        "Q": 8,
        "q": 8,
    }

    def __init__(self, fields: list[TelemetryField]) -> None:
        self.fields = fields

    def decommutate(self, payload: bytes) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for field in self.fields:
            size = self.TYPE_SIZES.get(field.data_type, 0)
            if field.offset + size > len(payload):
                continue

            fmt = f">{field.data_type}"
            raw_val = struct.unpack_from(fmt, payload, field.offset)[0]
            results[field.name] = (raw_val * field.scale) + field.offset_val

        return results

    @classmethod
    def from_dict(cls, config: list[dict[str, Any]]) -> DecomEngine:
        fields = [TelemetryField(**f) for f in config]
        return cls(fields)


__all__ = ["DecomEngine", "TelemetryField"]
