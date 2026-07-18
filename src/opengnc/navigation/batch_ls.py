"""
Differential Correction using Batch Least Squares for Orbit Determination.
"""


from typing import cast

import numpy as np

from opengnc.disturbances.gravity import TwoBodyGravity


class BatchLeastSquaresOD:
    r"""
    Differential Correction via Batch Least Squares for Orbit Determination.

    Iteratively refines the spacecraft state by minimizing weighted residuals
    over a batch of observations.
    Normal Equations: $(\mathbf{H}^T \mathbf{W} \mathbf{H}) \Delta \mathbf{x} = \mathbf{H}^T \mathbf{W} \mathbf{b}$.

    Parameters
    ----------
    x_guess : np.ndarray
        Initial 6D state guess $[\mathbf{r}, \mathbf{v}]^T$ (m, m/s).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default is Earth.
    """

    def __init__(self, x_guess: np.ndarray, mu: float = 398600.4415e9) -> None:
        """Initialize Batch LS solver."""
        self.x = np.asarray(x_guess, dtype=float)
        self.mu = mu
        self.gravity = TwoBodyGravity(mu=mu)

    def _propagate(self, x0: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        """
        Propagate state using RK4 integration.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state (6,).
        t_start, t_end : float
            Time interval (s).

        Returns
        -------
        np.ndarray
            State at $t_{end}$.
        """
        def f(state: np.ndarray) -> np.ndarray:
            r = state[:3]
            v = state[3:]
            a = self.gravity.get_acceleration(r)
            return np.concatenate([v, a])

        dt = t_end - t_start
        if abs(dt) < 1e-8:
            return cast(np.ndarray, x0)

        k1 = f(x0)
        k2 = f(x0 + 0.5 * dt * k1)
        k3 = f(x0 + 0.5 * dt * k2)
        k4 = f(x0 + dt * k3)
        return cast(np.ndarray, x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    def solve(
        self,
        observations: np.ndarray,
        times: np.ndarray,
        max_iter: int = 15,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Iteratively solve the Batch LS problem.

        Parameters
        ----------
        observations : np.ndarray
            Position observations $[r_x, r_y, r_z]$ (N, 3) (m).
        times : np.ndarray
            Observation timestamps (N,) (s).
        max_iter : int, optional
            Maximum iterations. Default 15.
        tol : float, optional
            Convergence tolerance. Default 1e-6.

        Returns
        -------
        np.ndarray
            Estimated state vector at $t=0$ (6,).
        """
        obs_vecs = np.asarray(observations)
        t_vecs = np.asarray(times)

        for _ in range(max_iter):
            a_list: list[np.ndarray] = []
            b_list: list[np.ndarray] = []

            for z, t in zip(obs_vecs, t_vecs):
                # 1. Prediction and Residual
                x_t = self._propagate(self.x, 0, t)
                z_hat = x_t[:3]
                b_list.append(z - z_hat)

                # 2. Linearized Mapper (Jacobian)
                r_vec = x_t[:3]
                r_mag = np.linalg.norm(r_vec)
                g_mat = (self.mu / r_mag**3) * (3.0 * np.outer(r_vec, r_vec) / r_mag**2 - np.eye(3))

                phi = np.eye(6)
                phi[:3, 3:] = np.eye(3) * t
                phi[3:, :3] = g_mat * t
                phi[3:, 3:] = np.eye(3) + g_mat * (t**2 / 2.0)

                h_mat = np.zeros((3, 6))
                h_mat[:, :3] = np.eye(3)
                a_list.append(h_mat @ phi)

            a_mat, b_vec = np.vstack(a_list), np.concatenate(b_list)

            # 3. Solve Normal Equations
            lhs, rhs = a_mat.T @ a_mat, a_mat.T @ b_vec
            try:
                dx = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dx = np.linalg.pinv(lhs) @ rhs

            self.x += dx
            if np.linalg.norm(dx) < tol:
                break

        return cast(np.ndarray, self.x)




