"""
Analytical Two-Body Propagator using Kepler's Equation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Propagator


class KeplerPropagator(Propagator):
    r"""
    Analytical Two-Body Propagator.

    Solves Kepler's Problem using Universal Variables:
    $M = E - e \sin E$

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    """

    def __init__(self, mu: float = 398600.4418e9) -> None:
        self.mu = mu

    def propagate(
        self, r_i: np.ndarray, v_i: np.ndarray, dt: float, n_iter: int = 100, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves the two-body problem using the Kepler equation (Vallado implementation).

        Parameters
        ----------
        r_i : np.ndarray
            Initial position vector (m).
        v_i : np.ndarray
            Initial velocity vector (m/s).
        dt : float
            Propagation duration (s).
        n_iter : int, optional
            Maximum number of iterations for Newton-Raphson. Default is 100.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r_f, v_f).
            r_f : Final position vector (m).
            v_f : Final velocity vector (m/s).
        """
        mu = self.mu
        r_i_mag = np.linalg.norm(r_i)
        v_i_mag = np.linalg.norm(v_i)
        energy = 0.5 * v_i_mag**2 - mu / r_i_mag  # Specific orbital energy

        alpha: float = float(-2 * energy / mu)  # Reciprocal of semi-major axis

        # Check for near-parabolic or undefined orbits
        if np.abs(energy) < 1e-9:
            alpha = 0.0

        # Universal variable initialization
        if alpha > 1e-9:  # Circular or Elliptical
            try:
                T = 2 * np.pi * np.sqrt(np.abs(1.0 / alpha) ** 3 / mu)  # Period
                if np.abs(dt) > np.abs(T):
                    dt = dt % T
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):  # pragma: no cover
                pass  # Fallback if calculation fails

            x_i = float(np.sqrt(mu) * dt * alpha)

        elif np.abs(alpha) < 1e-9:  # Parabolic
            h = np.cross(r_i, v_i)
            h_mag = np.linalg.norm(h)
            p = h_mag**2 / mu

            s = 0.5 * (np.pi / 2 - np.arctan(3 * np.sqrt(mu / (p**3)) * dt))
            w = np.arctan(np.power(np.tan(s), 1 / 3))

            x_i = float(np.sqrt(p) * (2 / np.tan(2 * w)))
            alpha = 0.0

        else:  # Hyperbolic
            # Avoid division by zero and domain errors
            try:
                temp = (
                    -2
                    * mu
                    * alpha
                    * dt
                    / (np.dot(r_i, v_i) + np.sign(dt) * np.sqrt(-mu / alpha) * (1 - r_i_mag * alpha))
                )
                x_i = float(np.sign(dt) * np.sqrt(-1 / alpha) * np.log(temp))
            except Exception:
                # Fallback for very specific hyperbolic cases or numerical issues
                x_i = float(np.sqrt(mu) * dt * alpha)  # Rough guess

        # Newton-Raphson iteration
        i = 0
        x_new = float(x_i)
        while i < n_iter:
            yaw = float((x_i**2) * alpha)
            c2, c3 = self._c23_eq(yaw)

            r_dot_v = np.dot(r_i, v_i)

            nom = (
                np.sqrt(mu) * dt
                - x_i**3 * c3
                - r_dot_v / np.sqrt(mu) * x_i**2 * c2
                - r_i_mag * x_i * (1 - yaw * c3)
            )
            denom = (
                x_i**2 * c2
                + r_dot_v / np.sqrt(mu) * x_i * (1 - yaw * c3)
                + r_i_mag * (1 - yaw * c2)
            )

            if abs(denom) < 1e-12:  # Avoid catastrophic cancellation
                break  # pragma: no cover

            x_new = x_i + nom / denom

            if np.abs(x_new - x_i) < 1e-8:
                break
            x_i = x_new
            i += 1

        c2, c3 = self._c23_eq(float((x_new**2) * alpha))
        f = 1 - x_new**2 / r_i_mag * c2
        g = dt - x_new**3 / np.sqrt(mu) * c3
        r_new = f * r_i + g * v_i
        r_new_mag = np.linalg.norm(r_new)

        fdot = np.sqrt(mu) / (r_new_mag * r_i_mag) * x_new * ((x_new**2) * alpha * c3 - 1)
        gdot = 1 - (x_new**2 / r_new_mag) * c2
        v_new = fdot * r_i + gdot * v_i

        return r_new, v_new

    def _c23_eq(self, yaw: float) -> tuple[float, float]:
        """
        Calculates Stumpff functions c2 and c3.

        Parameters
        ----------
        yaw : float
            Universal variable argument (alpha * x^2).

        Returns
        -------
        tuple[float, float]
            (c2, c3).
        """
        if yaw > 1e-8:
            sqrt_yaw = np.sqrt(yaw)
            c2 = (1 - np.cos(sqrt_yaw)) / yaw
            c3 = (sqrt_yaw - np.sin(sqrt_yaw)) / (yaw * sqrt_yaw)
        elif yaw < -1e-8:
            sqrt_neg_yaw = np.sqrt(-yaw)
            c2 = (1 - np.cosh(sqrt_neg_yaw)) / yaw
            c3 = (np.sinh(sqrt_neg_yaw) - sqrt_neg_yaw) / ((-yaw) * sqrt_neg_yaw)
        else:
            c2 = 0.5
            c3 = 1 / 6.0
        return float(c2), float(c3)




