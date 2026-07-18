"""
Covariance transformation and reachability analysis tools.
"""

from __future__ import annotations

from typing import cast

import numpy as np


class CovarianceTransform:
    """
    Tools for transforming state covariances between various frames.
    """

    @staticmethod
    def eci_to_ric(r_eci: np.ndarray, v_eci: np.ndarray, P_eci: np.ndarray) -> np.ndarray:
        """
        Transform covariance from ECI to RIC (Radial, In-Track, Cross-Track).

        Parameters
        ----------
        r_eci : np.ndarray
            Position (m).
        v_eci : np.ndarray
            Velocity (m/s).
        P_eci : np.ndarray
            $6 \times 6$ Covariance matrix in ECI.

        Returns
        -------
        np.ndarray
            $6 \times 6$ Covariance matrix in RIC.
        """
        # 1. Define RIC Unit Vectors
        u_r = r_eci / np.linalg.norm(r_eci)
        h = np.cross(r_eci, v_eci)
        u_c = h / np.linalg.norm(h)
        u_i = np.cross(u_c, u_r)

        # 2. Rotation Matrix R (ECI to RIC)
        rot = np.vstack([u_r, u_i, u_c])

        # 3. Full 6x6 Transformation Matrix T
        T = np.zeros((6, 6))
        T[0:3, 0:3] = rot
        T[3:6, 3:6] = rot

        # 4. Transform: P_ric = T P_eci T.T
        return cast(np.ndarray, T @ P_eci @ T.T)

    @staticmethod
    def ric_to_eci(r_eci: np.ndarray, v_eci: np.ndarray, P_ric: np.ndarray) -> np.ndarray:
        """
        Transform covariance from RIC back to ECI.

        Parameters
        ----------
        r_eci : np.ndarray
            Position (m).
        v_eci : np.ndarray
            Velocity (m/s).
        P_ric : np.ndarray
            $6 \times 6$ Covariance matrix in RIC.

        Returns
        -------
        np.ndarray
            $6 \times 6$ Covariance matrix in ECI.
        """
        u_r = r_eci / np.linalg.norm(r_eci)
        h = np.cross(r_eci, v_eci)
        u_c = h / np.linalg.norm(h)
        u_i = np.cross(u_c, u_r)

        rot = np.vstack([u_r, u_i, u_c]).T  # Inverse is Transpose

        T = np.zeros((6, 6))
        T[0:3, 0:3] = rot
        T[3:6, 3:6] = rot

        return cast(np.ndarray, T @ P_ric @ T.T)


class ReachabilityAnalysis:
    """
    Orbital maneuver reachability analysis.
    """

    def __init__(self, mu: float = 398600.4418e9) -> None:
        """Initialize with gravity constant."""
        self.mu = mu

    def get_reachable_delta_elements(
        self, a: float, e: float, i: float, dv_total: float
    ) -> dict[str, float]:
        r"""
        Calculate maximum possible changes in orbital elements for a total delta-V.

        Based on impulsive GVE approximations.

        Parameters
        ----------
        a, e, i : float
            Current semi-major axis, eccentricity, inclination.
        dv_total : float
            Total $\Delta V$ budget (m/s).

        Returns
        -------
        dict
            Max theoretical $\Delta a, \Delta e, \Delta i$.
        """
        v_circ = np.sqrt(self.mu / a)

        # Max delta-a: Apply all delta-V along the velocity vector (Hohmann-like)
        # dv = sqrt(mu/a) * (sqrt(2*r_next / (a+r_next)) - 1)
        # Simplified: da = 2 * a * dv / v_orbit
        max_da = 2 * a * dv_total / v_circ

        # Max delta-e: Apply at perigee/apogee
        max_de = 2 * dv_total / v_circ

        # Max delta-i: Apply at nodes
        # dv = 2 * v * sin(di/2)
        max_di = 2 * np.arcsin(dv_total / (2 * v_circ))

        return {
            "max_delta_a": max_da,
            "max_delta_e": max_de,
            "max_delta_i": np.degrees(max_di),
        }
