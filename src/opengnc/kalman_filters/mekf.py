"""
Multiplicative Extended Kalman Filter (MEKF) for Attitude Estimation.
"""

from __future__ import annotations

import numpy as np

from opengnc.utils.quat_utils import (
    quat_conj,
    quat_mult,
    quat_normalize,
    quat_rot,
    skew_symmetric,
)


class MEKF:
    """
    Multiplicative Extended Kalman Filter (MEKF) for Attitude Estimation.

    Maintains a global quaternion for orientation and an additive 3-component
    error vector in the tangent space for bias and local attitude corrections.

    Parameters
    ----------
    q_init : np.ndarray, optional
        Initial quaternion $[q_x, q_y, q_z, q_w]$. Defaults to identity $[0,0,0,1]$.
    beta_init : np.ndarray, optional
        Initial gyro bias $[b_x, b_y, b_z]$ (rad/s). Defaults to zeros.
    """

    def __init__(
        self,
        q_init: np.ndarray | None = None,
        beta_init: np.ndarray | None = None
    ) -> None:
        """Initialize reference state, error covariance, and noise matrices."""
        if q_init is None:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.q = quat_normalize(np.asarray(q_init))

        if beta_init is None:
            self.beta = np.zeros(3)
        else:
            self.beta = np.asarray(beta_init)

        # Covariance (6x6 for error state: [d_theta, d_beta])
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.001
        self.R = np.eye(3) * 0.01

        # Internal state vector (7x1) [quat, bias]
        self.x = np.concatenate([self.q, self.beta])

    def predict(self, omega_meas: np.ndarray, dt: float, q_mat: np.ndarray | None = None) -> None:
        """
        Predict the reference state and propagate error covariance.

        Parameters
        ----------
        omega_meas : np.ndarray
            Measured angular velocity in body frame (rad/s).
        dt : float
            Propagation interval (s).
        q_mat : np.ndarray, optional
            Process noise covariance (6x6). Defaults to `self.Q`.
        """
        qm = np.asarray(q_mat) if q_mat is not None else self.Q
        w_meas = np.asarray(omega_meas)

        # 1. Integrate Reference State: q_new = q_old * dq
        omega = w_meas - self.beta
        wm = np.linalg.norm(omega)

        if wm > 1e-10:
            axis = omega / wm
            angle = wm * dt
            dq = np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])
            self.q = quat_mult(self.q, dq)

        self.q = quat_normalize(self.q)

        # 2. Propagate Error Covariance: P = Phi * P * Phi' + Q_dt
        wx = skew_symmetric(omega)
        f_jac = np.zeros((6, 6))
        f_jac[0:3, 0:3] = -wx
        f_jac[0:3, 3:6] = -np.eye(3)

        phi = np.eye(6) + f_jac * dt
        self.P = (phi @ self.P @ phi.T) + (qm * dt)

        self.x = np.concatenate([self.q, self.beta])

    def update(
        self,
        z_body: np.ndarray,
        z_ref: np.ndarray,
        r_mat: np.ndarray | None = None
    ) -> None:
        """
        Perform a vector measurement update.

        Linearizes the observation of a reference vector (e.g., Sun, Earth)
        and applies a multiplicative correction to the quaternion.

        Parameters
        ----------
        z_body : np.ndarray
            Measured vector in spacecraft body frame (normalized).
        z_ref : np.ndarray
            Reference vector in inertial frame (normalized).
        r_mat : np.ndarray, optional
            Measurement noise covariance (3x3). Defaults to `self.R`.
        """
        r = np.asarray(r_mat) if r_mat is not None else self.R
        zb = np.asarray(z_body)
        zr = np.asarray(z_ref)

        # 1. Predicted observation: h(x) = C(q) * z_ref
        q_inv = quat_conj(self.q)
        zp = quat_rot(q_inv, zr)

        # 2. Sensitivity matrix: H = [ [zp]x | 0_{3x3} ]
        h_mat = np.zeros((3, 6))
        h_mat[:, 0:3] = skew_symmetric(zp)

        # 3. Kalman Gain
        s_mat = (h_mat @ self.P @ h_mat.T) + r
        k_gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)

        # 4. Correct error state
        dx = k_gain @ (zb - zp)
        dtheta = dx[0:3]
        dbeta = dx[3:6]

        # 5. Apply corrections
        # Quat Multiplicative: q = q * dq(dtheta/2)
        dq_corr = np.concatenate([0.5 * dtheta, [1.0]])
        self.q = quat_normalize(quat_mult(self.q, dq_corr))
        self.beta += dbeta

        # 6. Covariance Update (Joseph Form)
        i_kh = np.eye(6) - (k_gain @ h_mat)
        self.P = (i_kh @ self.P @ i_kh.T) + (k_gain @ r @ k_gain.T)

        self.x = np.concatenate([self.q, self.beta])




