"""
Symplectic integrator (Yoshida 4th order) for conservative systems.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator


class SymplecticIntegrator(Integrator):
    """
    Symplectic Integrator (Yoshida 4th order).
    Conserves Energy/Hamiltonian for conservative systems (like Two-Body gravity).
    Assumes state vector y = [r, v] where dy/dt = [v, a(r)].
    Specifically for systems where a depends only on r (a = f(r)).
    """

    def __init__(self) -> None:
        # Yoshida 4th Order Coefficients
        two_to_third = 2 ** (1 / 3)
        denom = 2 - two_to_third
        self.w1 = 1 / denom
        self.w0 = -two_to_third / denom

        self.c1 = self.w1 / 2
        self.c4 = self.c1
        self.c2 = (self.w0 + self.w1) / 2
        self.c3 = self.c2

        self.d1 = self.w1
        self.d3 = self.w1
        self.d2 = self.w0
        self.d4 = 0.0

    def integrate(
        self,
        f: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        dt: float = 10.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate over time span.
        Note: Symplectic methods work BEST for TIME-INVARIANT potentials (a = f(r)).
        f: function that returns [v, a].
        """
        t0, tf = t_span
        y = np.array(y0)
        h = dt

        if h > (tf - t0):
            h = tf - t0

        t_values = [t0]
        y_values = [y]

        curr_t = t0
        curr_y = y.copy()

        # Coefficients for step composition
        c = [self.c1, self.c2, self.c3, self.c4]
        d = [self.d1, self.d2, self.d3, 0.0]

        while curr_t < tf:
            if curr_t + h > tf:
                h = tf - curr_t

            # Yoshida 4th order step loop
            # 4 sub-steps of Position then Velocity update
            r = curr_y[:3]
            v = curr_y[3:]

            for i in range(3):
                # Update Position
                r = r + c[i] * h * v
                # Evaluate Acceleration with new position
                dy_sub = f(curr_t, np.concatenate([r, v]))
                a = dy_sub[3:]
                # Update Velocity
                v = v + d[i] * h * a

            # Fourth position update
            r = r + c[3] * h * v

            curr_y = np.concatenate([r, v])
            curr_t = curr_t + h

            t_values.append(curr_t)
            y_values.append(curr_y)

        return np.array(t_values), np.array(y_values)

    def step(
        self, f: Callable, t: float, y: np.ndarray, dt: float, **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Single step wrapper.
        """
        # Compose a single step
        res_t, res_y = self.integrate(f, (t, t + dt), y, dt=dt)
        return res_y[-1], res_t[-1], float(dt)
