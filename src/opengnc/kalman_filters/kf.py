"""
Standard Linear Kalman Filter (KF) implementation.
"""


from typing import cast

import numpy as np


class KF:
    """
    Standard Discrete-Time Linear Kalman Filter (KF).

    Suitable for linear estimation and navigation problems (e.g., constant
    velocity or constant acceleration models in Cartesian space).

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector $x$.
    dim_z : int
        Dimension of the measurement vector $z$.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """Initialize filter dimensions and default matrices."""
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.B: np.ndarray | None = None

    def predict(
        self,
        u: np.ndarray | None = None,
        f_mat: np.ndarray | None = None,
        q_mat: np.ndarray | None = None,
        b_mat: np.ndarray | None = None,
    ) -> None:
        """
        Predict the state and covariance one step forward.

        Equations:
        $x_{k|k-1} = F x_{k-1|k-1} + B u_k$
        $P_{k|k-1} = F P_{k-1|k-1} F^T + Q$

        Parameters
        ----------
        u : np.ndarray, optional
            Control input vector (dim_u,).
        f_mat : np.ndarray, optional
            State transition matrix (dim_x, dim_x). Defaults to `self.F`.
        q_mat : np.ndarray, optional
            Process noise covariance (dim_x, dim_x). Defaults to `self.Q`.
        b_mat : np.ndarray, optional
            Control input matrix (dim_x, dim_u). Defaults to `self.B`.
        """
        f = np.asarray(f_mat) if f_mat is not None else self.F
        q = np.asarray(q_mat) if q_mat is not None else self.Q
        b = np.asarray(b_mat) if b_mat is not None else self.B

        # 1. State Prediction
        if b is not None and u is not None:
            self.x = cast(np.ndarray, (f @ self.x) + (b @ np.asarray(u)))
        else:
            self.x = cast(np.ndarray, f @ self.x)

        # 2. Covariance Prediction
        self.P = (f @ self.P @ f.T) + q

    def update(
        self,
        z: np.ndarray,
        h_mat: np.ndarray | None = None,
        r_mat: np.ndarray | None = None
    ) -> None:
        """
        Update state estimate using a new measurement.

        Uses the Joseph robust form for covariance updates to maintain symmetry
        and positive-definiteness.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector (dim_z,).
        h_mat : np.ndarray, optional
            Measurement matrix (dim_z, dim_x). Defaults to `self.H`.
        r_mat : np.ndarray, optional
            Measurement noise covariance (dim_z, dim_z). Defaults to `self.R`.
        """
        h = np.asarray(h_mat) if h_mat is not None else self.H
        r = np.asarray(r_mat) if r_mat is not None else self.R
        zv = np.asarray(z)

        # 1. Innovation
        innov = zv - (h @ self.x)

        # 2. Innovation Covariance: S = HPH' + R
        s_mat = (h @ self.P @ h.T) + r

        # 3. Kalman Gain: K = PH' [S]^-1
        k_gain = self.P @ h.T @ np.linalg.inv(s_mat)

        # 4. State Correction
        self.x = cast(np.ndarray, self.x + (k_gain @ innov))

        # 5. Covariance Correction (Joseph Form): P = (I-KH)P(I-KH)' + KRK'
        i_mat = np.eye(self.dim_x)
        i_kh = i_mat - (k_gain @ h)
        self.P = (i_kh @ self.P @ i_kh.T) + (k_gain @ r @ k_gain.T)




