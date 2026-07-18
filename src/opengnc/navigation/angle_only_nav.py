"""
Navigation using Line-of-Sight (LOS) measurements (unit vectors).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .orbit_determination import OrbitDeterminationEKF


class AngleOnlyNavigation(OrbitDeterminationEKF):
    r"""
    Angles-Only Navigation (AON) using Line-of-Sight (LOS) unit vectors.

    Estimates the spacecraft state by tracking unit vectors to known celestial
    or terrestrial targets. Inherits dynamics from `OrbitDeterminationEKF`.

    Parameters
    ----------
    x_initial : np.ndarray
        Initial 6D state vector $[\mathbf{r}, \mathbf{v}]^T$ (m, m/s).
    p_initial : np.ndarray
        Initial estimation error covariance ($6\times 6$).
    q_mat : np.ndarray
        Process noise covariance ($6\times 6$).
    r_mat : np.ndarray
        Measurement noise covariance for the unit vector ($3\times 3$).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    """

    def update_unit_vector(self, u_meas: np.ndarray, target_pos_eci: np.ndarray) -> None:
        r"""
        Update the state estimate using a measured LOS unit vector $\mathbf{u}$.

        Measurement Model:
        $\mathbf{z} = \frac{\mathbf{r}_t - \mathbf{r}}{\rho} + \nu$
        Jacobian:
        $\mathbf{H}_r = -\frac{1}{\rho} (\mathbf{I} - \mathbf{u}\mathbf{u}^T)$

        Parameters
        ----------
        u_meas : np.ndarray
            Measured unit vector in ECI frame.
        target_pos_eci : np.ndarray
            Known coordinates of the target (m).
        """
        r_target = np.asarray(target_pos_eci)

        def hx(x: np.ndarray) -> np.ndarray:
            r = x[:3]
            rel_r = r_target - r
            rho = np.linalg.norm(rel_r)
            if rho < 1.0:
                return np.zeros(3)
            return cast(np.ndarray, rel_r / rho)

        def h_jac(x: np.ndarray) -> np.ndarray:
            r = x[:3]
            rel_r = r_target - r
            rho = np.linalg.norm(rel_r)

            if rho < 1.0:
                return np.zeros((3, 6))

            u = rel_r / rho
            # Projection matrix (I - uu^T)
            h_rel = cast(np.ndarray, -(np.eye(3) - np.outer(u, u)) / rho)

            h_mat = np.zeros((3, 6))
            h_mat[:, :3] = h_rel
            return h_mat

        self.ekf.update(np.asarray(u_meas), hx, h_jac)




