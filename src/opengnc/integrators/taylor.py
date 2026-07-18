"""
Taylor Series Numerical Integrator.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from opengnc.integrators.integrator import Integrator


class TaylorIntegrator(Integrator):
    r"""
    Taylor Series expansion-based numerical integrator.

    Solves $\dot{y} = f(t, y)$ by expanding $y$ as a Taylor series
    up to a specified order.

    Parameters
    ----------
    order : int, optional
        Taylor expansion order (1-4). Default 2.
    """

    def __init__(self, order: int = 2) -> None:
        """Initialize with expansion order."""
        self.order = order

    def step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        dt: float,
        **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Perform a single Taylor series step.

        Parameters
        ----------
        f : Callable
            Derivative function.
        t : float
            Current time.
        y : np.ndarray
            Current state.
        dt : float
            Time step.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (y_next, t_next, dt_suggested).
        """
        y_next = np.array(y, dtype=float)

        # 1st order term (Euler)
        f1 = f(t, y, **kwargs)
        y_next += f1 * dt

        if self.order >= 2:
            # 2nd order term
            # Approx f_dot = ∂f/∂t + ∂f/∂y * f
            # Using finite difference: (f(t+h, y+f*h) - f(t, y)) / h
            h = 1e-6
            f_next = f(t + h, y + f1 * h, **kwargs)
            f_dot = (f_next - f1) / h
            y_next += 0.5 * f_dot * dt**2

        if self.order >= 3:
            # 3rd order term approx
            h = 1e-6
            f_dot_next = (f(t + 2*h, y + f1*2*h, **kwargs) - f_next) / h
            f_ddot = (f_dot_next - f_dot) / h
            y_next += (1.0 / 6.0) * f_ddot * dt**3

        return y_next, t + dt, dt
