"""
Abstract base class for numerical integrators.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class Integrator(ABC):
    r"""
    Abstract base class for numerical ODE integrators.

    Provides a common interface for fixed-step and variable-step numerical
    integration of first-order differential equations $\dot{y} = f(t, y)$.
    """

    @abstractmethod
    def step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        dt: float,
        **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        r"""
        Perform a single integration step.

        Parameters
        ----------
        f : Callable
            Derivative function $f(t, y, \dots) \to \dot{y}$.
        t : float
            Current time $t_n$ (s).
        y : np.ndarray
            Current state $y_n$.
        dt : float
            User-requested time step size $h$ (s).
        **kwargs : Any
            Additional arguments for $f$.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (y_next, t_next, dt_suggested).
            - y_next: State at $t_n + dt$.
            - t_next: Final time $t_n + dt$ (s).
            - dt_suggested: Recommended next step size (s).
        """
        pass

    def integrate(
        self,
        f: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        dt: float = 0.01,
        **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Integrate the differential equation over a specified time interval.

        Parameters
        ----------
        f : Callable
            Derivative function $f(t, y, \dots)$.
        t_span : tuple[float, float]
            Interval of integration $(t_{start}, t_{final})$ (s).
        y0 : np.ndarray
            Initial state vector.
        dt : float, optional
            Initial time step size (s). Default 0.01.
        **kwargs : Any
            Additional arguments for $f$.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t_values, y_values) for documentation and plotting.
        """
        t0, tf = t_span
        t = float(t0)
        y = np.array(y0, dtype=float)

        t_values = [t]
        y_values = [y]

        curr_dt = float(dt)

        # Integration loop
        while t < tf:
            # Adjust step to hit end-time exactly
            if t + curr_dt > tf:
                curr_dt = tf - t

            y, t, curr_dt = self.step(f, t, y, curr_dt, **kwargs)

            t_values.append(t)
            y_values.append(y)

        return np.array(t_values), np.array(y_values)




