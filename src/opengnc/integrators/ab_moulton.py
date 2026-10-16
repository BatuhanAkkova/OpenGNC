"""
Adams-Bashforth-Moulton 8th order predictor-corrector integrator.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator
from .rk4 import RK4


class AdamsBashforthMoultonIntegrator(Integrator):
    """
    Adams-Bashforth-Moulton 8th order Integrator (Predictor-Corrector Ordinate Form).
    Treats the ODE system as a first-order system: dy/dt = f(t, y).
    """

    def __init__(self) -> None:
        # Predictor Coefficients (Denominator 120960)
        self.p_coeffs = (
            np.array([434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799])
            / 120960
        )

        # Corrector Coefficients (Denominator 120960)
        self.c_coeffs = (
            np.array([36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375]) / 120960
        )

    def integrate(
        self,
        f: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        dt: float = 10.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate over a time span using Adams-Bashforth-Moulton 8th order.
        """
        t0, tf = t_span
        y = np.array(y0)
        h = dt

        if h > (tf - t0):
            h = tf - t0

        t_values = [t0]
        y_values = [y]

        # Initialize with RK4 to fill 8 initial points of derivatives
        rk4 = RK4()

        history_dy = []  # History of state derivatives: dy/dt = f(t, y)
        curr_t = t0
        curr_y = y.copy()

        # Initial derivative
        history_dy.append(f(t0, curr_y))

        # Fill first 7 steps with RK4 (total 8 points including initial)
        for k in range(1, 8):
            curr_y, curr_t, _ = rk4.step(f, curr_t, curr_y, h)
            t_values.append(curr_t)
            y_values.append(curr_y)
            history_dy.append(f(curr_t, curr_y))

        # Propagation Loop from step 8 onwards
        while curr_t < tf:
            if curr_t + h > tf:
                h_last = tf - curr_t
                curr_y, curr_t, _ = rk4.step(f, curr_t, curr_y, h_last)
                t_values.append(curr_t)
                y_values.append(curr_y)
                break

            # Predictor Step: y^{p}_{n+1} = y_n + h * sum(p_j * dy_{n-j})
            sum_p = np.zeros_like(curr_y)
            for j in range(8):
                sum_p += self.p_coeffs[j] * history_dy[-(j + 1)]

            y_p = curr_y + h * sum_p
            next_t = curr_t + h
            dy_p = f(next_t, y_p)  # Evaluate predicted derivative

            # Corrector Step: y_{n+1} = y_n + h * sum(c_j * dy_{n+1-j})
            temp_history_dy = history_dy + [dy_p]
            sum_c = np.zeros_like(curr_y)
            for j in range(8):
                sum_c += self.c_coeffs[j] * temp_history_dy[-(j + 1)]

            y_c = curr_y + h * sum_c
            dy_c = f(next_t, y_c)  # Evaluate corrected derivative

            # Update step
            curr_y = y_c
            curr_t = next_t

            # Append correct derivative with latest states
            history_dy.append(dy_c)
            # Maintain 8 items history
            if len(history_dy) > 8:
                history_dy.pop(0)

            t_values.append(curr_t)
            y_values.append(curr_y)

        return np.array(t_values), np.array(y_values)

    def step(
        self, f: Callable, t: float, y: np.ndarray, dt: float, **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Single step interface (not recommended for multi-step integrators).
        """
        raise NotImplementedError(
            "Adams-Bashforth-Moulton requires historical states. Use integrate method."
        )
