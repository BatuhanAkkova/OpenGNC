"""
EKF for relative navigation using Clohessy-Wiltshire (Hill) dynamics.
"""


from typing import cast

import numpy as np

from opengnc.kalman_filters.ekf import EKF


class RelativeNavigationEKF:
    r"""
    Relative Navigation EKF via Clohessy-Wiltshire (Hill) Dynamics.

    Estimates the 6D relative state $\delta \mathbf{x} = [\delta \mathbf{r}, \delta \mathbf{v}]^T$
    in the Hill (RSW) frame. Assumes a circular target orbit.

    Parameters
    ----------
    x0 : np.ndarray
        Initial relative state $[x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$ (m, m/s).
    p0 : np.ndarray
        Initial estimation covariance ($6\times 6$).
    q_mat : np.ndarray
        Process noise covariance ($6\times 6$).
    r_mat : np.ndarray
        Measurement noise covariance ($3\times 3$).
    mean_motion : float
        Target mean motion $n = \sqrt{\mu/a^3}$ (rad/s).
    """

    def __init__(
        self,
        x0: np.ndarray,
        p0: np.ndarray,
        q_mat: np.ndarray,
        r_mat: np.ndarray,
        mean_motion: float
    ) -> None:
        """Initialize Relative EKF."""
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = cast(np.ndarray, np.asarray(x0, dtype=float))
        self.ekf.P = cast(np.ndarray, np.asarray(p0, dtype=float))
        self.ekf.Q = cast(np.ndarray, np.asarray(q_mat, dtype=float))
        self.ekf.R = cast(np.ndarray, np.asarray(r_mat, dtype=float))
        self.n = mean_motion

    def predict(self, dt: float) -> None:
        r"""
        Predict relative state using the analytical CW transition matrix $\Phi(t)$.

        Parameters
        ----------
        dt : float
            Predict step (s).
        """
        phi = self._get_cw_transition_matrix(self.n, dt)

        def fx(x: np.ndarray, dt_step: float, u: np.ndarray | None) -> np.ndarray:
            return cast(np.ndarray, phi @ x)

        def f_jac(x: np.ndarray, dt_step: float, u: np.ndarray | None) -> np.ndarray:
            return cast(np.ndarray, phi)

        self.ekf.predict(fx, f_jac, dt)

    def update(self, z_rel_pos: np.ndarray) -> None:
        """
        Update estimate using a 3D relative position measurement.

        Parameters
        ----------
        z_rel_pos : np.ndarray
            Relative position vector in Hill frame (m).
        """
        def hx(x: np.ndarray) -> np.ndarray:
            return cast(np.ndarray, x[:3])

        def h_jac(x: np.ndarray) -> np.ndarray:
            h_mat = np.zeros((3, 6))
            h_mat[:, :3] = np.eye(3)
            return cast(np.ndarray, h_mat)

        self.ekf.update(np.asarray(z_rel_pos), hx, h_jac)

    def _get_cw_transition_matrix(self, n: float, t: float) -> np.ndarray:
        r"""
        Compute the CW State Transition Matrix $\Phi(t)$.

        The Hill-frame geometry uses:
        - X: Radial (outward)
        - Y: Along-track (velocity direction)
        - Z: Cross-track (orbit normal)

        Parameters
        ----------
        n : float
            Mean motion.
        t : float
            Time.

        Returns
        -------
        np.ndarray
            $6\times 6$ transition matrix.
        """
        nt = n * t
        s, c = np.sin(nt), np.cos(nt)
        phi = np.zeros((6, 6))

        # Pos-Pos
        phi[0, 0], phi[0, 3] = 4 - 3*c, s/n
        phi[0, 4] = (2.0/n)*(1 - c)
        phi[1, 0], phi[1, 1] = 6*(s - nt), 1.0
        phi[1, 3], phi[1, 4] = (2.0/n)*(c - 1), (4.0*s/n) - 3*t
        phi[2, 2], phi[2, 5] = c, s/n

        # Vel-Pos/Vel
        phi[3, 0], phi[3, 3], phi[3, 4] = 3*n*s, c, 2*s
        phi[4, 0], phi[4, 3], phi[4, 4] = 6*n*(c - 1), -2*s, 4*c - 3
        phi[5, 2], phi[5, 5] = -n*s, c

        return cast(np.ndarray, phi)

    @property
    def state(self) -> np.ndarray:
        r"""Estimated relative state $[\delta \mathbf{r}, \delta \mathbf{v}]^T$."""
        return self.ekf.x




