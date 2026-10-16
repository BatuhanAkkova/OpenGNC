"""
Space Situational Awareness (SSA) Module.
"""

from .conjunction import compute_pc_chan, compute_pc_foster
from .maneuver import plan_avoidance_maneuver
from .tle_interface import TLECatalog, TLEEntity
from .tracking import compute_mahalanobis_distance, correlate_tracks

__all__ = [
    "TLECatalog",
    "TLEEntity",
    "compute_mahalanobis_distance",
    "compute_pc_chan",
    "compute_pc_foster",
    "correlate_tracks",
    "plan_avoidance_maneuver",
]
