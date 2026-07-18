"""
Fixed-step fourth-order Runge-Kutta integrator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator


class RK4(Integrator):
    r"""
    Fixed-step fourth-order Runge-Kutta integrator.

    The method evaluates four stage derivatives and combines them into a
    fourth-order accurate step update.
    """

    def step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray, float, float]:
        """Perform a single RK4 step."""
        y_val = np.asarray(y)
        k1 = np.asarray(f(t, y_val, **kwargs))
        k2 = np.asarray(f(t + 0.5 * dt, y_val + 0.5 * dt * k1, **kwargs))
        k3 = np.asarray(f(t + 0.5 * dt, y_val + 0.5 * dt * k2, **kwargs))
        k4 = np.asarray(f(t + dt, y_val + dt * k3, **kwargs))
        y_next = y_val + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_next, t + dt, dt
