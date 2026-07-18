"""
Adaptive-step Runge-Kutta-Fehlberg 4(5) integrator.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator


class RK45(Integrator):
    """
    Runge-Kutta-Fehlberg 4(5) Adaptive Variable Step Integrator.

    Implements the Fehlberg embedded method that calculates both a 4th and
    5th order solution at each step to estimate local truncation error.

    Parameters
    ----------
    rtol : float, optional
        Relative error tolerance. Default 1e-6.
    atol : float, optional
        Absolute error tolerance. Default 1e-9.
    safety_factor : float, optional
        Safety multiplier for step-size prediction. Default 0.9.
    min_factor : float, optional
        Minimum step-size reduction ratio. Default 0.2.
    max_factor : float, optional
        Maximum step-size expansion ratio. Default 10.0.
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 10.0,
    ) -> None:
        """Initialize RK45 coefficients and tolerances."""
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor

        # Fehlberg Tableau Coefficients
        self.c = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
        self.a = [
            [],
            [1 / 4],
            [3 / 32, 9 / 32],
            [1932 / 2197, -7200 / 2197, 7296 / 2197],
            [439 / 216, -8, 3680 / 513, -845 / 4104],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
        ]

        # y: 4-th order weights, z: 5-th order weights
        self.b4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
        self.b5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])

        self.E = self.b5 - self.b4  # Error estimation coefficients

    def step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        dt: float,
        **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Perform a single adaptive RK45 step with error control.

        Parameters
        ----------
        f : Callable
            Derivative function.
        t : float
            Current time $t_n$ (s).
        y : np.ndarray
            Current state.
        dt : float
            Trial time step size (s).
        **kwargs : Any
            Additional parameters for $f$.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (y_next, t_next, dt_new).

        Raises
        ------
        RuntimeError
            If convergence fails and step size falls below epsilon.
        """
        dt_current = float(dt)
        y_val = np.asarray(y)

        while True:
            # 1. Evaluate Stages
            k = []
            k.append(np.asarray(f(t, y_val, **kwargs)))

            for i in range(1, 6):
                sum_ak = np.zeros_like(y_val)
                for j in range(i):
                    sum_ak += self.a[i][j] * k[j]
                k.append(np.asarray(f(t + self.c[i] * dt_current, y_val + dt_current * sum_ak, **kwargs)))

            k_arr = np.array(k)

            # 2. Estimate solutions and error
            y_next = y_val + dt_current * (self.b5 @ k_arr)
            error_est = dt_current * (self.E @ k_arr)

            # 3. Compute error ratio
            scale = self.atol + self.rtol * np.maximum(np.abs(y_val), np.abs(y_next))
            error_ratio = np.max(np.abs(error_est) / scale)

            if error_ratio < 1.0:
                # Acceptance
                t_next = t + dt_current

                # Predict next optimal step size
                if error_ratio < 1e-10:
                    dt_new = dt_current * self.max_factor
                else:
                    dt_new = dt_current * self.safety_factor * (error_ratio**-0.2)

                dt_new = max(dt_current * self.min_factor, min(dt_new, dt_current * self.max_factor))
                return y_next, t_next, dt_new
            else:
                # Rejection
                dt_current = dt_current * self.safety_factor * (error_ratio**-0.25)
                if abs(dt_current) < 1e-15:
                    raise RuntimeError(f"RK45: Step size underflow at t={t}")

    def integrate(
        self,
        f: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        dt: float | None = None,
        **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate over span with adaptive stepping.

        Parameters
        ----------
        f : Callable
            Derivative function.
        t_span : tuple[float, float]
            (t0, tf) (s).
        y0 : np.ndarray
            Initial state.
        dt : float, optional
            Initial trial step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t_values, y_values).
        """
        if dt is None:
            dt = (t_span[1] - t_span[0]) / 100.0
        return super().integrate(f, t_span, y0, dt, **kwargs)




