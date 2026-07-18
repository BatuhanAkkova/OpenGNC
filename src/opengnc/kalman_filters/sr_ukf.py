"""
Square-root Unscented Kalman Filter (SR-UKF) algorithm.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular


class SRUKF:
    """
    Square-root Unscented Kalman Filter.

    Propagates the Cholesky factor of the covariance matrix for improved
    numerical stability relative to the standard UKF.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        self.gamma = np.sqrt(dim_x + self.lambda_)
        self.num_sigmas = 2 * dim_x + 1
        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        weight = 1.0 / (2.0 * (dim_x + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = weight
            self.Wc[i] = weight

        self.x = np.zeros(dim_x)
        self.S = np.eye(dim_x)
        self.Qs = np.eye(dim_x)
        self.Rs = np.eye(dim_z)

    def predict(
        self,
        dt: float,
        fx_func: Callable,
        qs_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Perform the SR-UKF predict step."""
        qs_curr = qs_mat if qs_mat is not None else self.Qs
        sigmas = self._generate_sigma_points()
        sigmas_f = np.zeros((self.num_sigmas, self.dim_x))
        for i in range(self.num_sigmas):
            sigmas_f[i] = fx_func(sigmas[i], dt, **kwargs)

        self.x = cast(np.ndarray, np.asarray(self.Wm @ sigmas_f, dtype=float))
        x_pts = np.sqrt(self.Wc[1]) * (sigmas_f[1:] - self.x)
        _, s_transpose = qr(np.vstack((x_pts, qs_curr)), mode="economic")
        self.S = s_transpose[: self.dim_x, : self.dim_x].T
        dx_vec = sigmas_f[0] - self.x
        self.S = self._cholesky_update(self.S, dx_vec, self.Wc[0])

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        rs_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Perform the SR-UKF update step."""
        rs_curr = rs_mat if rs_mat is not None else self.Rs
        sigmas_f = self._generate_sigma_points()
        sigmas_h = np.zeros((self.num_sigmas, self.dim_z))
        for i in range(self.num_sigmas):
            sigmas_h[i] = hx_func(sigmas_f[i], **kwargs)

        z_pred = self.Wm @ sigmas_h
        h_pts = np.sqrt(self.Wc[1]) * (sigmas_h[1:] - z_pred)
        _, s_transpose_y = qr(np.vstack((h_pts, rs_curr)), mode="economic")
        sy_mat = s_transpose_y[: self.dim_z, : self.dim_z].T
        dz_0 = sigmas_h[0] - z_pred
        sy_mat = self._cholesky_update(sy_mat, dz_0, self.Wc[0])

        pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.num_sigmas):
            dx_vec = sigmas_f[i] - self.x
            dz_vec = sigmas_h[i] - z_pred
            pxz += self.Wc[i] * np.outer(dx_vec, dz_vec)

        k_gain = solve_triangular(
            sy_mat,
            solve_triangular(sy_mat, pxz.T, lower=True),
            lower=True,
            trans="T",
        ).T
        self.x += k_gain @ (z - z_pred)

        u_mat = k_gain @ sy_mat
        for i in range(self.dim_z):
            self.S = self._cholesky_update(self.S, u_mat[:, i], -1.0)

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points from the current Cholesky factor."""
        sigmas = np.zeros((self.num_sigmas, self.dim_x))
        sigmas[0] = self.x
        for i in range(self.dim_x):
            sigmas[i + 1] = self.x + self.gamma * self.S[:, i]
            sigmas[i + 1 + self.dim_x] = self.x - self.gamma * self.S[:, i]
        return sigmas

    def _cholesky_update(self, s_mat: np.ndarray, vec: np.ndarray, weight: float) -> np.ndarray:
        """Perform a rank-1 Cholesky update or downdate."""
        if weight > 0:
            v_scaled = np.sqrt(weight) * vec
            _, s_transpose = qr(np.vstack((s_mat.T, v_scaled)), mode="economic")
            return np.asarray(s_transpose[: s_mat.shape[0], : s_mat.shape[1]].T, dtype=float)

        p_cov = s_mat @ s_mat.T + weight * np.outer(vec, vec)
        try:
            return np.asarray(cholesky(p_cov, lower=True), dtype=float)
        except np.linalg.LinAlgError:
            return np.asarray(
                cholesky(p_cov + np.eye(s_mat.shape[0]) * 1e-12, lower=True),
                dtype=float,
            )

    @property
    def P(self) -> np.ndarray:
        """Return the full covariance matrix ``P = S S^T``."""
        return np.asarray(self.S @ self.S.T, dtype=float)
