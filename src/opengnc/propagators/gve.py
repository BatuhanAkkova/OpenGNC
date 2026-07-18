"""
Gauss-Variational Equations (GVE) Propagator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from opengnc.propagators.base import Propagator


class GVEPropagator(Propagator):
    """
    Propagator based on Gauss-Variational Equations.

    Suitable for analyzing the effects of small perturbations on
    Keplerian elements.

    Parameters
    ----------
    mu : float
        Gravitational parameter (m^3/s^2).
    """

    def __init__(self, mu: float = 398600.4418e9) -> None:
        """Initialize with gravity constant."""
        self.mu = mu

    def propagate(  # type: ignore[override]
        self,
        t0: float,
        state0: np.ndarray,
        dt: float,
        perturbation_func: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        r"""
        Propagate orbital elements one step using RK4 on GVE.

        Parameters
        ----------
        t0 : float
            Initial time (s).
        state0 : np.ndarray
            Initial Keplerian elements $[a, e, i, \Omega, \omega, \theta]$.
        dt : float
            Time step (s).
        perturbation_func : Callable
            Function returning acceleration vector $[a_r, a_s, a_w]$ in RIC frame.

        Returns
        -------
        np.ndarray
            New orbital elements.
        """
        # Runge-Kutta 4th order integration
        k1 = self.gve_derivatives(t0, state0, perturbation_func)
        k2 = self.gve_derivatives(t0 + 0.5 * dt, state0 + 0.5 * dt * k1, perturbation_func)
        k3 = self.gve_derivatives(t0 + 0.5 * dt, state0 + 0.5 * dt * k2, perturbation_func)
        k4 = self.gve_derivatives(t0 + dt, state0 + dt * k3, perturbation_func)

        return np.asarray(state0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    def gve_derivatives(
        self,
        t: float,
        state: np.ndarray,
        perturbation_func: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        r"""
        Calculate GVE state derivatives.

        Parameters
        ----------
        t : float
            Time.
        state : np.ndarray
            $[a, e, i, \Omega, \omega, \theta]$.
        perturbation_func : Callable
            RIC acceleration function.

        Returns
        -------
        np.ndarray
            $[da, de, di, d\Omega, d\omega, d\theta]$.
        """
        a, e, i, raan, argp, nu = state

        # Avoid division by zero for circular/equatorial orbits
        e = max(e, 1e-12)
        i = max(i, 1e-12)

        p = a * (1 - e**2)
        h = np.sqrt(self.mu * p)
        r = p / (1 + e * np.cos(nu))
        u = argp + nu

        # Get perturbations in RIC frame
        # a_r: Radial, a_s: In-Track (transverse), a_w: Cross-Track (normal)
        acc_ric = perturbation_func(t, state)
        ar, as_, aw = acc_ric

        # GVE Equations
        da = (2 * a**2 / h) * (e * np.sin(nu) * ar + (p / r) * as_)
        de = (1 / h) * (p * np.sin(nu) * ar + ((p + r) * np.cos(nu) + r * e) * as_)
        di = (r * np.cos(u) / h) * aw
        draan = (r * np.sin(u) / (h * np.sin(i))) * aw
        dargp = (1 / (h * e)) * (-p * np.cos(nu) * ar + (p + r) * np.sin(nu) * as_) - \
                (r * np.sin(u) * np.cos(i) / (h * np.sin(i))) * aw
        dnu = (h / r**2) + (1 / (h * e)) * (p * np.cos(nu) * ar - (p + r) * np.sin(nu) * as_)

        return np.array([da, de, di, draan, dargp, dnu])
