"""
Deprecated compatibility wrapper for the experimental EDL namespace.
"""

from opengnc._compat import warn_legacy_import
from opengnc.experimental.edl import (
    aerocapture_guidance,
    ballistic_entry_dynamics,
    calculate_g_load,
    hazard_avoidance,
    lifting_entry_dynamics,
    sutton_grave_heating,
)

warn_legacy_import("opengnc.edl", "opengnc.experimental.edl")

__all__ = [
    "aerocapture_guidance",
    "ballistic_entry_dynamics",
    "calculate_g_load",
    "hazard_avoidance",
    "lifting_entry_dynamics",
    "sutton_grave_heating",
]
