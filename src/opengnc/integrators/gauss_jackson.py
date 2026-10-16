"""
Gauss-Jackson 8th order predictor-corrector integrator for second-order ODEs.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator
from .rk4 import RK4


class GaussJacksonIntegrator(Integrator):
    """
    Gauss-Jackson 8th order Integrator (Predictor-Corrector Summed Form).
    Specifically for second-order ODEs: dy/dt = [v, a].
    Assumes state y = [r, v] where r, v are 3D vectors.
    """

    def __init__(self) -> None:
        # Predictor Coefficients (Position) - Full series from index 0
        self.p_pos = np.array(
            [
                1,
                0,
                1 / 12,
                1 / 12,
                19 / 240,
                3 / 40,
                863 / 12096,
                275 / 4032,
                33953 / 518400,
                8183 / 129600,
            ]
        )

        # Predictor Coefficients (Velocity)
        self.p_vel = np.array(
            [
                1,
                1 / 2,
                5 / 12,
                3 / 8,
                251 / 720,
                95 / 288,
                19087 / 60480,
                5257 / 17280,
                1070017 / 3628800,
                25713 / 89600,
            ]
        )

        # Corrector Coefficients (Position)
        self.c_pos = np.array(
            [
                1,
                -1,
                1 / 12,
                0,
                -1 / 240,
                -1 / 240,
                -221 / 60480,
                -19 / 6048,
                -9829 / 3628800,
                -407 / 172800,
            ]
        )

        # Corrector Coefficients (Velocity)
        self.c_vel = np.array(
            [
                1,
                1 / 2,
                1 / 12,
                1 / 24,
                19 / 720,
                3 / 160,
                863 / 60480,
                275 / 24192,
                33953 / 3628800,
                8183 / 1036800,
            ]
        )

    def _calc_differences(self, history: list[np.ndarray]) -> np.ndarray:
        r"""
        Calculate backward differences \nabla^j a_n.
        history is a list of N elements [a_0, a_1, ..., a_n] where index -1 is current.
        Returns array of differences at current step: [a_n, \nabla a_n, \nabla^2 a_n, ...]
        """
        N = len(history)
        if N == 0:
            return np.zeros((1, 3))

        diff = np.zeros((N, history[0].shape[0]))
        diff[0, :] = history[-1]  # \nabla^0 a_n = a_n

        current_diff = np.array(history)
        for j in range(1, N):
            current_diff = current_diff[1:] - current_diff[:-1]
            diff[j, :] = current_diff[-1]

        return diff

    def integrate(
        self,
        f: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        dt: float = 10.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate over a time span using Gauss-Jackson 8th order.
        """
        t0, tf = t_span
        y = np.array(y0)
        h = dt

        if h > (tf - t0):
            h = tf - t0

        t_values = [t0]
        y_values = [y]

        # Initialize with single-step method (RK4)
        rk4 = RK4()

        history_r = [y[:3]]
        history_v = [y[3:]]
        # Calculate initial acceleration
        dy0 = f(t0, y)
        history_a = [dy0[3:]]

        curr_t = t0
        curr_y = y.copy()

        # We need 8 points (n=0 to 7)
        for k in range(1, 8):
            curr_y, curr_t, _ = rk4.step(f, curr_t, curr_y, h)
            t_values.append(curr_t)
            y_values.append(curr_y)

            history_r.append(curr_y[:3])
            history_v.append(curr_y[3:])
            dy = f(curr_t, curr_y)
            history_a.append(dy[3:])

        # Compute Back Differences at step 7
        diffs = self._calc_differences(history_a)

        # Initialize Sum Variables S1, S2 at step 7
        sum_c_vel = np.zeros(3)
        sum_c_pos = np.zeros(3)
        for j in range(8):
            sum_c_vel += self.c_vel[j] * diffs[j]
            sum_c_pos += self.c_pos[j] * diffs[j]

        S1 = history_v[-1] / h - sum_c_vel
        S2 = history_r[-1] / (h**2) - sum_c_pos

        curr_t = t_values[-1]
        curr_y = y_values[-1]

        print(f"DEBUG INIT: S1={S1}, S2={S2}")

        # Propagation Loop from step 8 onwards
        step_count = 8
        while curr_t < tf:
            if curr_t + h > tf:
                h_last = tf - curr_t
                curr_y, curr_t, _ = rk4.step(f, curr_t, curr_y, h_last)
                t_values.append(curr_t)
                y_values.append(curr_y)
                break

            # Predictor step for step n+1
            diffs_n = self._calc_differences(history_a)

            sum_p_vel = np.zeros(3)
            sum_p_pos = np.zeros(3)
            for j in range(8):
                sum_p_vel += self.p_vel[j] * diffs_n[j]
                sum_p_pos += self.p_pos[j] * diffs_n[j]

            v_p = h * (S1 + sum_p_vel)
            r_p = (h**2) * (S2 + S1 + sum_p_pos)

            if step_count < 12:
                print(f"STEP {step_count} PREDICTOR:")
                print(f"  r_p: {r_p}")
                print(f"  v_p: {v_p}")

            # Evaluate Accelerations
            y_p = np.concatenate([r_p, v_p])
            next_t = curr_t + h
            dy_p = f(next_t, y_p)
            a_p = dy_p[3:]

            # Corrector Step
            n_iter = 3
            v_c = v_p.copy()
            r_c = r_p.copy()
            a_c_iter = a_p.copy()

            for _ in range(n_iter):
                # Update history with latest acceleration estimate
                temp_history_a = history_a[1:] + [a_c_iter]
                diffs_next = self._calc_differences(temp_history_a)

                S1_next = S1 + a_c_iter
                S2_next = S2 + S1_next

                sum_c_vel_next = np.zeros(3)
                sum_c_pos_next = np.zeros(3)
                for j in range(8):
                    sum_c_vel_next += self.c_vel[j] * diffs_next[j]
                    sum_c_pos_next += self.c_pos[j] * diffs_next[j]

                v_c = h * (S1_next + sum_c_vel_next)
                r_c = (h**2) * (S2_next + sum_c_pos_next)

                # Evaluate acceleration with corrected state
                y_c = np.concatenate([r_c, v_c])
                dy_c = f(next_t, y_c)
                a_c_iter = dy_c[3:]

            a_c = a_c_iter

            if step_count < 12:
                print(f"STEP {step_count} CORRECTOR:")
                print(f"  r_c: {r_c}")
                print(f"  v_c: {v_c}")
                print(f"  a_c: {a_c}")

            # Final update of sums
            S1 = S1 + a_c
            S2 = S2 + S1

            # Update history_a
            history_a = history_a[1:] + [a_c]

            curr_t = next_t
            curr_y = y_c.copy()

            t_values.append(curr_t)
            y_values.append(curr_y)

            step_count += 1

        # Convert back to arrays to maintain interface consistency
        return np.array(t_values), np.array(y_values)

    def step(
        self, f: Callable, t: float, y: np.ndarray, dt: float, **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Single step interface.
        """
        raise NotImplementedError("Gauss-Jackson requires historical states. Use integrate method.")
