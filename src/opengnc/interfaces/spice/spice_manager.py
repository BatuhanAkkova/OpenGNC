"""
SPICE kernel and frame management for STK/SPICE interoperability.
"""

from pathlib import Path
from typing import cast

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
        self.loaded_kernels: list[str] = []

    def load_kernel(self, kernel_path: str) -> None:
        """Load a SPICE kernel such as ``.bsp``, ``.tpc``, or ``.tls``."""
        if not SPICEYPY_AVAILABLE:
            raise ImportError(
                "spiceypy is required for SPICE manager. Install via 'pip install OpenGNC[spk]'"
            )

        kernel = Path(kernel_path)
        if not kernel.exists():
            raise FileNotFoundError(f"Kernel not found: {kernel_path}")

        spice.furnsh(str(kernel))
        self.loaded_kernels.append(str(kernel))

    def clear_kernels(self) -> None:
        """Unload all kernels."""
        if SPICEYPY_AVAILABLE:
            spice.kclear()
            self.loaded_kernels = []

    def utc_to_et(self, utc_str: str) -> float:
        """Convert a UTC string to ephemeris time."""
        if not SPICEYPY_AVAILABLE:
            raise ImportError(
                "spiceypy is required for SPICE manager. Install via 'pip install OpenGNC[spk]'"
            )
        return float(spice.str2et(utc_str))

    def get_state(
        self, target: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> np.ndarray:
        """
        Get the state vector of a target body.

        Returns
        -------
        np.ndarray
            ``[x, y, z, vx, vy, vz]`` in meters and meters per second.
        """
        if not SPICEYPY_AVAILABLE:
            raise ImportError(
                "spiceypy is required for SPICE manager. Install via 'pip install OpenGNC[spk]'"
            )

        state, _ = spice.spkezr(target, et, frame, "NONE", observer)
        return cast(np.ndarray, np.asarray(state, dtype=float) * 1000.0)

    def transform_state(
        self, state: np.ndarray, et: float, from_frame: str, to_frame: str
    ) -> np.ndarray:
        """Transform a state vector between reference frames."""
        if not SPICEYPY_AVAILABLE:
            raise RuntimeError("spiceypy required.")

        xform = np.asarray(spice.sxform(from_frame, to_frame, et), dtype=float)
        transformed_state = xform @ np.asarray(state, dtype=float)
        return cast(np.ndarray, np.asarray(transformed_state, dtype=float))
