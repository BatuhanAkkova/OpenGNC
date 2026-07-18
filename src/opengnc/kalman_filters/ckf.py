"""
Cubature Kalman Filter (CKF) using spherical-radial rule for non-linear estimation.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.linalg import cholesky


class CKF:
    """
    Cubature Kalman Filter (CKF) using the spherical-radial rule.

    Offers superior numerical stability and accuracy for high-dimensional
    non-linear systems compared to the UKF. Uses exactly $2n$ cubature points
    with equal weights.

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector $x$.
    dim_z : int
        Dimension of the measurement vector $z$.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """Initialize CKF weights and unit cubature points."""
        self.dim_x = dim_x
        self.dim_z = dim_z

        # 1. Spherical-Radial weights: w_i = 1 / (2*n)
        self.num_points = 2 * dim_x
        self.weight = 1.0 / self.num_points

        # 2. Unit cubature points: xi_i = [sqrt(n) * e_i, -sqrt(n) * e_i]
        self.xi = np.zeros((dim_x, self.num_points))
        scale = np.sqrt(dim_x)
        for i in range(dim_x):
            self.xi[i, i] = scale
            self.xi[i, i + dim_x] = -scale

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(
        self,
        dt: float,
        fx_func: Callable,
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Cubature predict step.

        Parameters
        ----------
        dt : float
            Propagation time step (s).
        fx_func : Callable
            Non-linear transition function $f(x, dt, \dots) \to x_{new}$.
        q_mat : np.ndarray, optional
            Process noise covariance. Defaults to `self.Q`.
        **kwargs : Any
            Additional parameters for $f$.
        """
        q = np.asarray(q_mat) if q_mat is not None else self.Q

        # 1. Generate points using Cholesky of P
        points = self._generate_cubature_points(self.x, self.P)

        # 2. Propagate points through non-linear transition
        points_f = np.zeros((self.num_points, self.dim_x))
        for i in range(self.num_points):
            points_f[i] = fx_func(points[:, i], dt, **kwargs)

        # 3. Predicted mean
        self.x = self.weight * np.sum(points_f, axis=0)

        # 4. Predicted covariance
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.num_points):
            dx = points_f[i] - self.x
            self.P += self.weight * np.outer(dx, dx)
        self.P += q

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Cubature update step.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector.
        hx_func : Callable
            Non-linear measurement model $h(x, \dots) \to z_{pred}$.
        r_mat : np.ndarray, optional
            Measurement noise covariance. Defaults to `self.R`.
        **kwargs : Any
            Additional parameters for $h$.
        """
        r = np.asarray(r_mat) if r_mat is not None else self.R
        zv = np.asarray(z)

        # 1. Regenerate cubature points
        points_f = self._generate_cubature_points(self.x, self.P)

        # 2. Transform to measurement space
        points_h = np.zeros((self.num_points, self.dim_z))
        for i in range(self.num_points):
            points_h[i] = hx_func(points_f[:, i], **kwargs)

        # 3. Measurement mean
        zp = self.weight * np.sum(points_h, axis=0)

        # 4. Cross-covariance and Innovation covariance
        s_mat = np.zeros((self.dim_z, self.dim_z))
        pxz = np.zeros((self.dim_x, self.dim_z))

        for i in range(self.num_points):
            dx = points_f[:, i] - self.x
            dz = points_h[i] - zp
            s_mat += self.weight * np.outer(dz, dz)
            pxz += self.weight * np.outer(dx, dz)

        s_mat += r

        # 5. Kalman Gain and state/covariance correction
        k_gain = pxz @ np.linalg.inv(s_mat)
        self.x += k_gain @ (zv - zp)
        self.P -= k_gain @ s_mat @ k_gain.T

    def _generate_cubature_points(self, x: np.ndarray, p_cov: np.ndarray) -> np.ndarray:
        """
        Generate $2n$ points using the Cholesky factor of the covariance matrix.

        Parameters
        ----------
        x : np.ndarray
            Current mean vector.
        p_cov : np.ndarray
            Current error covariance matrix.

        Returns
        -------
        np.ndarray
            Points array (dim_x, num_points).
        """
        # Ensure symmetry and PDness for Cholesky
        p_sym = (p_cov + p_cov.T) / 2 + np.eye(self.dim_x) * 1e-12
        try:
            chol_l = cholesky(p_sym, lower=True)
            return cast(np.ndarray, x[:, np.newaxis] + (chol_l @ self.xi))
        except np.linalg.LinAlgError:
            # Fallback to sqrtm if Cholesky fails due to semi-definiteness
            from scipy.linalg import sqrtm
            sqrt_p = sqrtm(p_sym).real
            return cast(np.ndarray, x[:, np.newaxis] + (sqrt_p @ self.xi))




