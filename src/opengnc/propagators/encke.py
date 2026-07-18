"""
Encke's method propagator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ..integrators.integrator import Integrator
from ..integrators.rk4 import RK4
from .base import Propagator
from .kepler import KeplerPropagator


class EnckePropagator(Propagator):
    """
    Encke's method propagator.

    Integrates the deviation from a reference Keplerian orbit and is suitable
    for orbit propagation with small disturbances.

    Parameters
    ----------
    integrator : Integrator, optional
        Numerical integrator for the deviation states. Defaults to ``RK4``.
    mu : float, optional
        Gravitational parameter in cubic meters per square second.
    rect_tol : float, optional
        Rectification tolerance based on ``norm(dr) / norm(r_ref)``.
    """

    def __init__(
        self,
        integrator: Integrator | None = None,
        mu: float = 398600.4418e9,
        rect_tol: float = 1e-6,
    ) -> None:
        self.integrator = integrator if integrator else RK4()
        self.mu = mu
        self.rect_tol = rect_tol
        self.kepler = KeplerPropagator(mu=mu)

    def propagate(
        self,
        r_i: np.ndarray,
        v_i: np.ndarray,
        dt: float,
        perturbation_acc_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a state using Encke's method."""
        step_size = float(kwargs.get("dt_step", 10.0))
        if step_size > dt:
            step_size = dt

        curr_t = 0.0
        curr_r_ref = np.array(r_i, dtype=float)
        curr_v_ref = np.array(v_i, dtype=float)
        curr_y_dev = np.zeros(6, dtype=float)

        while curr_t < dt:
            h = step_size
            if curr_t + h > dt:
                h = dt - curr_t

            def local_eom(t_local: float, y: np.ndarray) -> np.ndarray:
                d_r = y[:3]
                d_v = y[3:]
                r_r, v_r = self.kepler.propagate(curr_r_ref, curr_v_ref, t_local)
                r_r_mag = np.linalg.norm(r_r)
                r_tot = r_r + d_r
                v_tot = v_r + d_v
                q_val = np.dot(d_r, d_r - 2.0 * r_r) / (r_r_mag**2)

                with np.errstate(divide="ignore", invalid="ignore"):
                    if abs(q_val) < 1e-12:
                        f_q_val = 0.0
                    else:
                        f_q_val = (
                            q_val * (3.0 + 3.0 * q_val + q_val**2) / (1.0 + (1.0 + q_val) ** 1.5)
                        )

                a_pt = np.zeros(3, dtype=float)
                if perturbation_acc_fn is not None:
                    a_pt = np.asarray(
                        perturbation_acc_fn(curr_t + t_local, r_tot, v_tot), dtype=float
                    )

                a_en = a_pt + (self.mu / (r_r_mag**3)) * (f_q_val * r_r - d_r)
                return cast(np.ndarray, np.concatenate([d_v, a_en]))

            next_y_dev, _, _ = self.integrator.step(local_eom, 0.0, curr_y_dev, h)
            curr_t += h

            curr_r_ref_next, curr_v_ref_next = self.kepler.propagate(curr_r_ref, curr_v_ref, h)
            dr_next = next_y_dev[:3]
            r_tot_next = curr_r_ref_next + dr_next
            r_tot_next_mag = np.linalg.norm(r_tot_next)

            if np.linalg.norm(dr_next) / r_tot_next_mag > self.rect_tol:
                curr_r_ref = r_tot_next
                curr_v_ref = curr_v_ref_next + next_y_dev[3:]
                curr_y_dev = np.zeros(6, dtype=float)
            else:
                curr_r_ref = curr_r_ref_next
                curr_v_ref = curr_v_ref_next
                curr_y_dev = next_y_dev

        r_f = curr_r_ref + curr_y_dev[:3]
        v_f = curr_v_ref + curr_y_dev[3:]
        return cast(np.ndarray, r_f), cast(np.ndarray, v_f)
