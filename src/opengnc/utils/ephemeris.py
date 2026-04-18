"""
JPL Ephemeris integration using jplephem.
"""

import os
from typing import Optional, cast

import numpy as np

try:
    from jplephem.spk import SPK
    JPLEPHEM_AVAILABLE = True
except ImportError:
    JPLEPHEM_AVAILABLE = False


class JPLEphemeris:
    """
    Handler for JPL SPK (Space Property Kernel) files.

    Provides high-precision position and velocity for planets, Sun, and Moon.

    Parameters
    ----------
    spk_path : str, optional
        Path to the .bsp SPK file (e.g., de421.bsp).
    """

    def __init__(self, spk_path: Optional[str] = None) -> None:
        """Initialize SPK kernel."""
        self.kernel = None
        if not JPLEPHEM_AVAILABLE:
            return

        if spk_path and os.path.exists(spk_path):
            self.kernel = SPK.open(spk_path)
        else:
            # Look for default in package directory if available
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "de421.bsp")
            if os.path.exists(default_path):
                self.kernel = SPK.open(default_path)

    def get_position(self, body_id: int, jd: float) -> np.ndarray:
        """
        Get body position in km (usually relative to solar system barycenter).

        Parameters
        ----------
        body_id : int
            JPL Body ID (e.g., 399 for Earth, 301 for Moon, 10 for Sun).
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in meters.
        """
        if not JPLEPHEM_AVAILABLE:
            raise ImportError(
                "jplephem is required for JPL Ephemeris. Install via 'pip install OpenGNC[spk]'"
            )

        if self.kernel is None:
            raise RuntimeError("SPK Kernel not loaded. Provide a valid .bsp file.")

        # jplephem returns km
        pos_km = self.kernel[0, body_id].compute(jd)
        return cast(np.ndarray, pos_km * 1000.0)

    def get_state(self, body_id: int, jd: float) -> np.ndarray:
        """
        Get body position and velocity.

        Parameters
        ----------
        body_id : int
            JPL Body ID.
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            State vector [x, y, z, vx, vy, vz] in m, m/s.
        """
        if not JPLEPHEM_AVAILABLE:
            raise ImportError(
                "jplephem is required for JPL Ephemeris. Install via 'pip install OpenGNC[spk]'"
            )

        if self.kernel is None:
            raise RuntimeError("SPK Kernel not loaded.")

        # compute_and_differentiate returns [x, y, z, vx, vy, vz] in km, km/day
        state_km = self.kernel[0, body_id].compute_and_differentiate(jd)
        pos = state_km[:3] * 1000.0
        vel = state_km[3:] * 1000.0 / 86400.0
        return cast(np.ndarray, np.concatenate([pos, vel]))
