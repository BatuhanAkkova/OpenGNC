"""
Extended Kalman Filter (EKF) for non-linear systems using Jacobians.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


class EKF:
    """
    Extended Kalman Filter (EKF) for non-linear systems.

    Linearizes the non-linear state transition and measurement models around the
    current estimate using first-order Taylor expansion (Jacobians).

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector $x$.
    dim_z : int
        Dimension of the measurement vector $z$.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """
        Initialize filter dimensions and initial matrices.

        Parameters
        ----------
        dim_x : int
            Dimension of state vector.
        dim_z : int
            Dimension of measurement vector.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(
        self,
        fx_func: Callable[..., np.ndarray],
        f_jac_func: Callable[..., np.ndarray],
        dt: float,
        u: np.ndarray | None = None,
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Non-linear state prediction.

        Equations:
        - State Predict: $\mathbf{\hat{x}}_{k|k-1} = f(\mathbf{\hat{x}}_{k-1|k-1}, \Delta t, \mathbf{u})$
        - Covariance Predict: $\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$
        where $\mathbf{F}$ is the state transition Jacobian.

        Parameters
        ----------
        fx_func : Callable
            Transition function.
        f_jac_func : Callable
            Jacobian of fx_func.
        dt : float
            Propagation step (s).
        u : np.ndarray | None, optional
            Control input.
        q_mat : np.ndarray | None, optional
            Process noise.
        **kwargs : Any
            Additional parameters.
        """
        q = np.asarray(q_mat) if q_mat is not None else self.Q

        # 1. Non-linear state propagation
        self.x = fx_func(self.x, dt, u, **kwargs)

        # 2. Linearized covariance propagation
        f_mat = f_jac_func(self.x, dt, u, **kwargs)
        self.P = (f_mat @ self.P @ f_mat.T) + q

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable[..., np.ndarray],
        h_jac_func: Callable[..., np.ndarray],
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Non-linear measurement update.

        Equations:
        - Innovation: $\mathbf{y} = \mathbf{z} - h(\mathbf{\hat{x}}_{k|k-1})$
        - Gain: $\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$
        - Update: $\mathbf{\hat{x}}_{k|k} = \mathbf{\hat{x}}_{k|k-1} + \mathbf{K} \mathbf{y}$
        - Joseph Form Covariance: $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K} \mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$

        Parameters
        ----------
        z : np.ndarray
            Measurement vector.
        hx_func : Callable
            Measurement model.
        h_jac_func : Callable
            Jacobian of hx_func.
        r_mat : np.ndarray | None, optional
            Measurement noise.
        **kwargs : Any
            Additional parameters.
        """
        r = np.asarray(r_mat) if r_mat is not None else self.R
        zv = np.asarray(z)

        # 1. Innovation using non-linear model
        resid = zv - hx_func(self.x, **kwargs)

        # 2. Linearized sensitivity matrix
        h_mat = h_jac_func(self.x, **kwargs)

        # 3. Innovation covariance
        s_mat = (h_mat @ self.P @ h_mat.T) + r

        # 4. Kalman Gain
        k_gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)

        # 5. Correct state and covariance (Joseph Form)
        self.x = self.x + (k_gain @ resid)

        i_kh = np.eye(self.dim_x) - (k_gain @ h_mat)
        self.P = (i_kh @ self.P @ i_kh.T) + (k_gain @ r @ k_gain.T)




