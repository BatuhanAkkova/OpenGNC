"""
Finite-Horizon Linear Quadratic Regulator (LQR).
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.integrate import solve_ivp


class FiniteHorizonLQR:
    r"""
    Finite-Horizon Linear Quadratic Regulator (LQR).

    Computes a time-varying optimal control law $u(t) = -K(t)x(t)$ for systems
    over a fixed time interval $[0, T]$.

    Objective: $\min J = x(T)^T P_f x(T) + \int_{0}^{T} (x^T Q(t) x + u^T R(t) u) dt$

    Parameters
    ----------
    A_fn : Callable[[float], np.ndarray]
        Time-varying state matrix $A(t)$ (nx x nx).
    B_fn : Callable[[float], np.ndarray]
        Time-varying input matrix $B(t)$ (nx x nu).
    Q_fn : Callable[[float], np.ndarray]
        Time-varying state cost matrix $Q(t)$ (nx x nx).
    R_fn : Callable[[float], np.ndarray]
        Time-varying input cost matrix $R(t)$ (nu x nu).
    Pf : np.ndarray
        Terminal state cost matrix at $t=T$ (nx x nx).
    T : float
        The fixed final time of the optimization horizon.
    """

    def __init__(
        self,
        A_fn: Callable[[float], np.ndarray],
        B_fn: Callable[[float], np.ndarray],
        Q_fn: Callable[[float], np.ndarray],
        R_fn: Callable[[float], np.ndarray],
        Pf: np.ndarray,
        T: float,
    ) -> None:
        """Initialize the Finite-Horizon LQR problem."""
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.Q_fn = Q_fn
        self.R_fn = R_fn
        self.Pf = np.asarray(Pf)
        self.T = float(T)
        self.P_trajectory: np.ndarray | None = None
        self.t_span: np.ndarray | None = None

    def solve(self, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Solve the Differential Riccati Equation (DRE) backwards in time.

        $-\dot{P} = P A + A^T P - P B R^{-1} B^T P + Q$, with $P(T) = P_f$

        Parameters
        ----------
        num_points : int, optional
            Number of points for the solution trajectory. Default is 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of (t_span, P_trajectory) where P_trajectory is (num_points, nx, nx).
        """
        nx = self.Pf.shape[0]

        def dre(t: float, p_flat: np.ndarray) -> np.ndarray:
            p_mat = p_flat.reshape((nx, nx))
            a_cur = self.A_fn(t)
            b_cur = self.B_fn(t)
            q_cur = self.Q_fn(t)
            r_cur = self.R_fn(t)

            # Solve for R^-1 * B.T @ P numerically
            k_term = np.linalg.solve(r_cur, b_cur.T @ p_mat)
            p_dot = -(p_mat @ a_cur + a_cur.T @ p_mat - p_mat @ b_cur @ k_term + q_cur)
            return cast(np.ndarray, p_dot.flatten())

        # Backward integration from T to 0
        t_eval = np.linspace(self.T, 0, num_points)
        sol = solve_ivp(dre, [self.T, 0], self.Pf.flatten(), t_eval=t_eval, method="RK45")

        # Reverse results to be monotonic from 0 to T
        self.t_span = cast(np.ndarray, sol.t[::-1])
        self.P_trajectory = cast(np.ndarray, sol.y.T[::-1].reshape((-1, nx, nx)))

        return self.t_span, self.P_trajectory

    def get_gain(self, t: float) -> np.ndarray:
        """
        Compute/interpolate the optimal feedback gain K at time t.

        Parameters
        ----------
        t : float
            Current time (s).

        Returns
        -------
        np.ndarray
            Feedback gain matrix K (nu x nx).
        """
        if self.P_trajectory is None or self.t_span is None:
            self.solve()
        if self.P_trajectory is None or self.t_span is None:
            raise RuntimeError("Riccati trajectory not available.")

        # Linearly interpolate each element of the P matrix
        nx = self.Pf.shape[0]
        p_t = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                p_t[i, j] = np.interp(t, self.t_span, self.P_trajectory[:, i, j])

        b_cur = self.B_fn(t)
        r_cur = self.R_fn(t)
        return cast(np.ndarray, np.linalg.solve(r_cur, b_cur.T @ p_t))

    def compute_control(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute optimal control input $u = -K(t)x$.

        Parameters
        ----------
        x : np.ndarray
            Current state vector (nx,).
        t : float
            Current time (s).

        Returns
        -------
        np.ndarray
            Optimal control input u (nu,).
        """
        k_t = self.get_gain(t)
        return cast(np.ndarray, -k_t @ np.asarray(x))
