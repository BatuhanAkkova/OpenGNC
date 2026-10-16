"""
Numerical Cowell Propagator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ..integrators.integrator import Integrator
from ..integrators.rk4 import RK4
from .base import Propagator


class CowellPropagator(Propagator):
    """
    Numerical Cowell Propagator.

    Integrates the equations of motion numerically, allowing for perturbations.

    Parameters
    ----------
    integrator : Integrator, optional
        Numerical integrator instance. Defaults to RK4.
    mu : float, optional
        Gravitational parameter (m^3/s^2). Default is Earth's.
    """

    def __init__(self, integrator: Integrator | None = None, mu: float = 398600.4418e9) -> None:
        self.integrator = integrator if integrator else RK4()
        self.mu = mu

    def propagate(
        self,
        r_i: np.ndarray,
        v_i: np.ndarray,
        dt: float,
        perturbation_acc_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagates the state using numerical integration.

        Parameters
        ----------
        r_i : np.ndarray
            Initial position vector (m).
        v_i : np.ndarray
            Initial velocity vector (m/s).
        dt : float
            Propagation duration (s).
        perturbation_acc_fn : callable, optional
            Function that returns perturbation accelerations.
            Signature: acc_pert = f(t, r, v) -> np.ndarray.
        **kwargs : dict
            Additional arguments, e.g., 'dt_step' for integration step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r_f, v_f).
            r_f : Final position vector (m).
            v_f : Final velocity vector (m/s).
        """
        # Define the state vector y = [r, v]
        y0 = np.concatenate([r_i, v_i])

        # Define the equations of motion: dy/dt = f(t, y)
        def equations_of_motion(t: float, y: np.ndarray) -> np.ndarray:
            r = y[:3]
            v = y[3:]
            r_mag = np.linalg.norm(r)

            # Two-body acceleration
            a_two_body = -self.mu / (r_mag**3) * r

            # Perturbations
            a_pert = np.zeros(3)
            if perturbation_acc_fn:
                a_pert = perturbation_acc_fn(t, r, v)

            a_total = a_two_body + a_pert

            return cast(np.ndarray, np.concatenate([v, a_total]))

        # Determine integration step size
        # If the user provides 'dt_step' in kwargs, use it.
        # Integrator.integrate takes an initial 'dt'.
        step_size = kwargs.get("dt_step", 10.0)  # Default 10s step size for numerical integration
        if step_size > dt:
            step_size = dt

        # Integrate
        # We start at t=0 and go to t=dt relative to the epoch of r_i
        t_span = (0.0, float(dt))
        t_values, y_values = self.integrator.integrate(
            equations_of_motion, t_span, y0, dt=step_size
        )

        # Extract final state
        y_final = y_values[-1]
        r_f = y_final[:3]
        v_f = y_final[3:]

        return cast(np.ndarray, r_f), cast(np.ndarray, v_f)
