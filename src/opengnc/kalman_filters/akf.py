"""
Adaptive Kalman Filter (AKF) with online covariance estimation (Myers-Tapley).
"""

from typing import cast

import numpy as np

from opengnc.kalman_filters.kf import KF


class AKF(KF):
    """
    Adaptive Kalman Filter (AKF) using Myers-Tapley online covariance estimation.

    Estimates process noise covariance (Q) and measurement noise covariance (R)
    online using the innovation sequence within a moving window.

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the measurement vector.
    window_size : int, optional
        Moving window size (N) for covariance estimation. Default is 20.
    """

    def __init__(self, dim_x: int, dim_z: int, window_size: int = 20) -> None:
        super().__init__(dim_x, dim_z)
        self.window_size = window_size
        self.innovations: list[np.ndarray] = []
        self.h_list: list[np.ndarray] = []
        self.p_minus_list: list[np.ndarray] = []
        self.fpf_t_list: list[np.ndarray] = []
        self.dx_list: list[np.ndarray] = []
        self.p_plus_list: list[np.ndarray] = []

    def predict(
        self,
        u: np.ndarray | None = None,
        f_mat: np.ndarray | None = None,
        q_mat: np.ndarray | None = None,
        b_mat: np.ndarray | None = None,
    ) -> None:
        """
        Predict step (stores history for adaptation).

        Parameters
        ----------
        u : np.ndarray, optional
            Control input vector.
        f_mat : np.ndarray, optional
            State transition matrix.
        q_mat : np.ndarray, optional
            Process noise covariance.
        b_mat : np.ndarray, optional
            Control input matrix.
        """
        f_curr = f_mat if f_mat is not None else self.F

        # Store F * P_{k-1|k-1} * F^T for Q estimation
        self.fpf_t_list.append(f_curr @ self.P @ f_curr.T)

        super().predict(u, f_mat, q_mat, b_mat)

        # Store P_{k|k-1} for R estimation
        self.p_minus_list.append(self.P.copy())

        if len(self.p_minus_list) > self.window_size:
            self.p_minus_list.pop(0)
            self.fpf_t_list.pop(0)

    def update(
        self, z: np.ndarray, h_mat: np.ndarray | None = None, r_mat: np.ndarray | None = None
    ) -> None:
        """
        Update step with online R and Q adaptation.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector.
        h_mat : np.ndarray, optional
            Measurement matrix.
        r_mat : np.ndarray, optional
            Measurement noise covariance.
        """
        h_curr = h_mat if h_mat is not None else self.H

        x_minus = self.x.copy()
        innov = z - (h_curr @ self.x)

        super().update(z, h_mat, r_mat)

        dx = self.x - x_minus

        self.innovations.append(innov)
        self.h_list.append(h_curr)
        self.dx_list.append(dx)
        self.p_plus_list.append(self.P.copy())

        if len(self.innovations) > self.window_size:
            self.innovations.pop(0)
            self.h_list.pop(0)
            self.dx_list.pop(0)
            self.p_plus_list.pop(0)

        # Perform adaptation if we have enough samples
        if len(self.innovations) >= self.window_size:
            self._adapt_noise_covariances()

    def _adapt_noise_covariances(self) -> None:
        """Estimates Q and R based on the history of innovations and state corrections."""
        n_val = self.window_size
        sum_yyt = np.zeros((self.dim_z, self.dim_z))
        sum_hpht = np.zeros((self.dim_z, self.dim_z))
        sum_q = np.zeros((self.dim_x, self.dim_x))

        for i in range(n_val):
            y_innov = self.innovations[i]
            h_mat = self.h_list[i]
            p_minus = self.p_minus_list[i]
            dx_corr = self.dx_list[i]
            p_plus = self.p_plus_list[i]
            fpf_t = self.fpf_t_list[i]

            sum_yyt += np.outer(y_innov, y_innov)
            sum_hpht += h_mat @ p_minus @ h_mat.T

            # Myers-Tapley formula for Q
            sum_q += np.outer(dx_corr, dx_corr) + p_plus - fpf_t

        r_hat = (1.0 / n_val) * sum_yyt - (1.0 / n_val) * sum_hpht
        self.R = self._make_psd(r_hat)

        q_hat = (1.0 / n_val) * sum_q
        self.Q = self._make_psd(q_hat)

    def _make_psd(self, matrix: np.ndarray) -> np.ndarray:
        """
        Force a matrix to be positive semi-definite by clipping negative eigenvalues.

        Parameters
        ----------
        matrix : np.ndarray
            Input symmetric matrix.

        Returns
        -------
        np.ndarray
            PSD matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        return cast(np.ndarray, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)




