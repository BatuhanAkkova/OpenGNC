"""
SGP4/SDP4 Analytical Propagator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sgp4.api import Satrec
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

from .base import Propagator


class Sgp4Propagator(Propagator):
    """
    SGP4/SDP4 Analytical Propagator.

    Wraps the `sgp4` library to propagate orbits using Two-Line Elements (TLEs).
    State output is typically in the TEME frame.

    Parameters
    ----------
    line1 : str
        First line of the TLE.
    line2 : str
        Second line of the TLE.
    """

    def __init__(self, line1: str, line2: str) -> None:
        if not SGP4_AVAILABLE:
            raise ImportError(
                "sgp4 is required for Sgp4Propagator. Install via 'pip install OpenGNC[sat]'"
            )
        self.sat = Satrec.twoline2rv(line1, line2)
        # TLE Epoch Julian Date
        self.jdsatepoch = self.sat.jdsatepoch
        self.jdsatepochF = self.sat.jdsatepochF

    def propagate(
        self, r_i: np.ndarray, v_i: np.ndarray, dt: float, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagates the satellite state from TLE epoch forward by dt.

        Note: initial position/velocity input (r_i, v_i) is IGNORED,
        as SGP4 is fully determined by the initialized TLE.

        Parameters
        ----------
        r_i : np.ndarray
            Ignored initial position.
        v_i : np.ndarray
            Ignored initial velocity.
        dt : float
            Propagation time relative to TLE epoch (s).
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r_f, v_f).
            r_f : Position in TEME frame (m).
            v_f : Velocity in TEME frame (m/s).
        """
        # SGP4 takes time in minutes from epoch
        minutes_from_epoch = dt / 60.0

        # sgp4 evaluation
        # returns (error_code, position, velocity)
        # position in km, velocity in km/s (TEME)
        err, r_km, v_kms = self.sat.sgp4(
            self.jdsatepoch, self.jdsatepochF + minutes_from_epoch / 1440.0
        )

        if err != 0:
            raise RuntimeError(f"SGP4 error code {err}: Propagation failed.")

        # Convert to SI units [m, m/s]
        r_f = np.array(r_km) * 1000.0
        v_f = np.array(v_kms) * 1000.0

        return r_f, v_f

    def propagate_to_jd(self, jd_f: float, jd_f_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate to a specific Julian Date.

        Parameters
        ----------
        jd_f : float
            Final Julian Date (integer part).
        jd_f_frac : float, optional
            Final Julian Date (fractional part). Default is 0.0.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r_f, v_f) in TEME frame (m, m/s).

        Raises
        ------
        RuntimeError
            If SGP4 propagation fails.
        """
        err, r_km, v_kms = self.sat.sgp4(jd_f, jd_f_frac)
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err}")
        return np.array(r_km) * 1000.0, np.array(v_kms) * 1000.0




