"""
Ensemble Kalman Filter (EnKF) using Monte Carlo samples for covariance representation.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np


class EnKF:
    """
    Ensemble Kalman Filter.

    Uses an ensemble of states to represent the error covariance matrix.
    """

    def __init__(self, dim_x: int, dim_z: int, ensemble_size: int = 50) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_ensemble = ensemble_size
        self.X = np.zeros((dim_x, self.num_ensemble))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def initialize_ensemble(self, x_mean: np.ndarray, p_cov: np.ndarray) -> None:
        """Initialize the ensemble from a multivariate normal distribution."""
        self.X = cast(
            np.ndarray,
            np.random.multivariate_normal(x_mean, p_cov, self.num_ensemble).T,
        )

    def predict(
        self,
        dt: float,
        fx_func: Callable,
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Propagate each ensemble member forward in time."""
        q_curr = q_mat if q_mat is not None else self.Q
        for i in range(self.num_ensemble):
            self.X[:, i] = cast(np.ndarray, fx_func(self.X[:, i], dt, **kwargs))
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), q_curr)
            self.X[:, i] += noise

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Update the ensemble using a measurement."""
        r_curr = r_mat if r_mat is not None else self.R
        z_ensemble = np.zeros((self.dim_z, self.num_ensemble))
        for i in range(self.num_ensemble):
            z_ensemble[:, i] = cast(np.ndarray, hx_func(self.X[:, i], **kwargs))

        z_mean = np.mean(z_ensemble, axis=1, keepdims=True)
        x_mean_vec = np.mean(self.X, axis=1, keepdims=True)
        anomalies_x = self.X - x_mean_vec
        anomalies_z = z_ensemble - z_mean

        z_perturbed = np.zeros((self.dim_z, self.num_ensemble))
        for i in range(self.num_ensemble):
            noise = np.random.multivariate_normal(np.zeros(self.dim_z), r_curr)
            z_perturbed[:, i] = z + noise

        innov_ensemble = z_perturbed - z_ensemble
        s_mat = (1.0 / (self.num_ensemble - 1)) * (anomalies_z @ anomalies_z.T) + r_curr
        pxz = (1.0 / (self.num_ensemble - 1)) * (anomalies_x @ anomalies_z.T)
        k_gain = pxz @ np.linalg.inv(s_mat)
        self.X += k_gain @ innov_ensemble

    @property
    def x(self) -> np.ndarray:
        """Return the ensemble mean state."""
        return cast(np.ndarray, np.mean(self.X, axis=1))

    @property
    def P(self) -> np.ndarray:
        """Return the ensemble covariance matrix."""
        anomalies_x = self.X - np.mean(self.X, axis=1, keepdims=True)
        return cast(
            np.ndarray,
            (1.0 / (self.num_ensemble - 1)) * (anomalies_x @ anomalies_x.T),
        )
