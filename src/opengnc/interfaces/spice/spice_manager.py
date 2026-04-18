"""
SPICE Kernel and Frame management for STK/SPICE interoperability.
"""

import os
from typing import List, Optional

import numpy as np

try:
    import spiceypy as spice
    SPICEYPY_AVAILABLE = True
except ImportError:
    SPICEYPY_AVAILABLE = False


class SpiceManager:
    """
    Standardized interface for SPICE kernels and frame transformations.
    Wraps SpiceyPy to provide STK-compatible environment modeling.
    """

    def __init__(self) -> None:
        self.loaded_kernels: List[str] = []
        if not SPICEYPY_AVAILABLE:
            print("Warning: spiceypy not installed. SPICE functionality will be disabled.")

    def load_kernel(self, kernel_path: str) -> None:
        """Load a SPICE kernel (.bsp, .tpc, .tls, etc.)."""
        if not SPICEYPY_AVAILABLE:
            return
        if os.path.exists(kernel_path):
            spice.furnsh(kernel_path)
            self.loaded_kernels.append(kernel_path)
        else:
            raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    def clear_kernels(self) -> None:
        """Unload all kernels."""
        if SPICEYPY_AVAILABLE:
            spice.kclear()
            self.loaded_kernels = []

    def utc_to_et(self, utc_str: str) -> float:
        """Convert UTC string to Ephemeris Time (ET)."""
        if not SPICEYPY_AVAILABLE:
            raise RuntimeError("spiceypy required.")
        return float(spice.str2et(utc_str))

    def get_state(
        self, target: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> np.ndarray:
        """
        Get state vector of a target body.

        Returns
        -------
        np.ndarray
            [x, y, z, vx, vy, vz] in meters and m/s.
        """
        if not SPICEYPY_AVAILABLE:
            raise RuntimeError("spiceypy required.")

        state, lt = spice.spkezr(target, et, frame, "NONE", observer)
        # state is in km and km/s
        return np.array(state) * 1000.0

    def transform_state(
        self, state: np.ndarray, et: float, from_frame: str, to_frame: str
    ) -> np.ndarray:
        """Transform state vector between frames."""
        if not SPICEYPY_AVAILABLE:
            raise RuntimeError("spiceypy required.")

        xform = spice.sxform(from_frame, to_frame, et)
        transformed_state = np.dot(xform, state)
        return transformed_state
