"""
Atmospheric wind and earth co-rotation modeling.
"""

import numpy as np


class AtmosphereCoRotation:
    """
    Atmospheric Wind Model (Strict Co-Rotation).

    Assumes that the atmosphere rotates perfectly in sync with the planet's
    Earth-Centered Earth-Fixed (ECEF) frame.

    Parameters
    ----------
    omega_e : float, optional
        Planetary angular rate (rad/s). Default Earth WGS84 rate.
    """

    def __init__(self, omega_e: float = 7.2921151467e-5) -> None:
        """Initialize Earth rotation vector."""
        self.omega_earth = np.array([0.0, 0.0, float(omega_e)])

    def get_wind_velocity(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        r"""
        Calculate local wind velocity vector in ECI.

        Equation:
        $\mathbf{v}_w = \boldsymbol{\omega}_E \times \mathbf{r}_{eci}$

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            Wind velocity vector (m/s).
        """
        rv = np.asarray(r_eci)
        return np.asarray(np.cross(self.omega_earth, rv))

    def get_relative_velocity(
        self,
        r_eci: np.ndarray,
        v_eci: np.ndarray,
        jd: float
    ) -> np.ndarray:
        r"""
        Calculate spacecraft velocity relative to the atmosphere (Airspeed).

        Equation:
        $\mathbf{v}_{rel} = \mathbf{v}_{sc} - \mathbf{v}_w$

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position (m).
        v_eci : np.ndarray
            ECI spacecraft velocity (m/s).
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            Relative velocity vector (m/s).
        """
        rv = np.asarray(r_eci)
        vv = np.asarray(v_eci)
        v_wind = self.get_wind_velocity(rv, jd)
        return np.asarray(vv - v_wind)




