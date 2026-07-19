"""
Deprecated compatibility wrapper for operational de-commutation helpers.
"""

from opengnc._compat import warn_legacy_import
from opengnc.ops.ground_segment.decom import DecomEngine, TelemetryField

warn_legacy_import("opengnc.ground_segment.decom", "opengnc.ops.ground_segment.decom")

__all__ = ["DecomEngine", "TelemetryField"]
