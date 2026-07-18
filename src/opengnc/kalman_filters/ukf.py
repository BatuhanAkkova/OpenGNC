"""
Unscented Kalman Filter (UKF) with support for states on manifolds.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import cholesky, sqrtm

from opengnc.utils.quat_utils import axis_angle_to_quat, quat_conj, quat_mult, quat_normalize


class UKF:
    r"""
    Generalized Unscented Kalman Filter (UKF).

    Propagates state and covariance through non-linear functions using the
    Unscented Transform (UT) with support for manifolds (e.g., $S^3$ for quaternions).

    Parameters
    ----------
    dim_x : int
        Dimension of the full state vector $x$.
    dim_z : int
        Dimension of the measurement vector $z$.
    dim_p : int, optional
        Dimension of the covariance $P$ in tangent space. Defaults to `dim_x`.
    alpha : float, optional
        UT tuning parameter $(10^{-4}, 1)$. Controls spread of sigma points.
    beta : float, optional
        UT parameter incorporating prior knowledge of distribution. Default 2.0.
    kappa : float, optional
        UT secondary scaling parameter. Usually 0 or $3-L$.
    subtract_x : Callable, optional
        Tangent space difference $(x_1, x_2) \to dx$.
    add_x : Callable, optional
        Manifold state update $(x, dx) \to x_{new}$.
    mean_x : Callable, optional
        Weighted manifold mean $(sigmas, weights) \to x_{mean}$.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dim_p: int | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        subtract_x: Callable[..., np.ndarray] | None = None,
        add_x: Callable[..., np.ndarray] | None = None,
        mean_x: Callable[..., np.ndarray] | None = None,
    ) -> None:
        r"""
        Initialize UKF with Unscented Transform parameters.

        Parameters
        ----------
        dim_x : int
            Full state vector dimension.
        dim_z : int
            Measurement vector dimension.
        dim_p : int | None, optional
            Error covariance dimension (tangent space).
        alpha : float, optional
            Sigma point spread tuning. Default 1e-3.
        beta : float, optional
            Distribution prior. Default 2.0 (Gaussian).
        kappa : float, optional
            Secondary scaling. Default 0.
        subtract_x : Callable | None, optional
            Error calculation $(x_1, x_2) \to dx$.
        add_x : Callable | None, optional
            State update $(x, dx) \to x_{new}$.
        mean_x : Callable | None, optional
            Weighted manifold mean calculation.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_p = dim_p if dim_p is not None else dim_x

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # 1. UT Scaling Parameters
        # $\lambda = \alpha^2 (\text{dim_p} + \kappa) - \text{dim_p}$
        self.lambda_ = alpha**2 * (self.dim_p + kappa) - self.dim_p
        # $\gamma = \sqrt{\text{dim_p} + \lambda}$
        self.gamma = np.sqrt(self.dim_p + self.lambda_)
        self.num_sigmas = 2 * self.dim_p + 1

        # 2. Weights for Mean (m) and Covariance (c)
        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)

        # $W_m^{(0)} = \frac{\lambda}{\text{dim_p} + \lambda}$
        self.Wm[0] = self.lambda_ / (self.dim_p + self.lambda_)
        # $W_c^{(0)} = \frac{\lambda}{\text{dim_p} + \lambda} + (1 - \alpha^2 + \beta)$
        self.Wc[0] = self.lambda_ / (self.dim_p + self.lambda_) + (1 - alpha**2 + beta)

        # $W_m^{(i)} = W_c^{(i)} = \frac{1}{2(\text{dim_p} + \lambda)}$ for $i=1, \dots, 2\text{dim_p}$
        w = 1.0 / (2 * (self.dim_p + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = w
            self.Wc[i] = w

        # 3. Default manifold operations (fall back to vector space)
        self.subtract_x = subtract_x if subtract_x is not None else lambda x1, x2: x1 - x2
        self.add_x = add_x if add_x is not None else lambda x, dx: x + dx
        self.mean_x = (
            mean_x if mean_x is not None else lambda sigmas, weights: np.dot(weights, sigmas)
        )

        self.x = np.zeros(dim_x)
        self.P = np.eye(self.dim_p)
        self.Q = np.eye(self.dim_p)
        self.R = np.eye(dim_z)

    def predict(
        self,
        dt: float,
        fx_func: Callable[..., np.ndarray],
        q_mat: np.ndarray | None = None,
        **kwargs: Any
    ) -> None:
        """
        Unscented transform prediction step.

        Parameters
        ----------
        dt : float
            Time step (s).
        fx_func : Callable
            Non-linear transition function.
        q_mat : np.ndarray | None, optional
            Process noise covariance.
        **kwargs : Any
            Additional parameters.
        """
        q = np.asarray(q_mat) if q_mat is not None else self.Q

        # 1. Generate sigma points in tangent space
        sigmas = self.generate_sigma_points(self.x, self.P)

        # 2. Transform sigma points
        sigmas_f = []
        for i in range(self.num_sigmas):
            sigmas_f.append(fx_func(sigmas[i], dt, **kwargs))
        sigmas_f_arr = np.array(sigmas_f)

        # 3. Calculate predicted mean
        self.x = self.mean_x(sigmas_f_arr, self.Wm)

        # 4. Calculate predicted covariance
        self.P = np.zeros((self.dim_p, self.dim_p))
        for i in range(self.num_sigmas):
            dx = self.subtract_x(sigmas_f_arr[i], self.x)
            self.P += self.Wc[i] * np.outer(dx, dx)
        self.P += q * dt

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any
    ) -> None:
        r"""
        Unscented update step.

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

        # 1. Regenerate sigma points from current prior
        sigmas_f = self.generate_sigma_points(self.x, self.P)

        # 2. Transform to observation space
        sigmas_h = []
        for i in range(self.num_sigmas):
            sigmas_h.append(hx_func(sigmas_f[i], **kwargs))
        sigmas_h_arr = np.array(sigmas_h)

        # 3. Compute measurement mean and covariances
        zp = np.dot(self.Wm, sigmas_h_arr)

        s_mat = np.zeros((self.dim_z, self.dim_z))
        pxz = np.zeros((self.dim_p, self.dim_z))

        for i in range(self.num_sigmas):
            dz = sigmas_h_arr[i] - zp
            dx = self.subtract_x(sigmas_f[i], self.x)

            s_mat += self.Wc[i] * np.outer(dz, dz)
            pxz += self.Wc[i] * np.outer(dx, dz)

        s_mat += r

        # 4. Correct state and covariance
        k_gain = pxz @ np.linalg.inv(s_mat)
        self.x = self.add_x(self.x, k_gain @ (zv - zp))
        self.P = self.P - (k_gain @ s_mat @ k_gain.T)

    def generate_sigma_points(self, x: np.ndarray, p_cov: np.ndarray) -> np.ndarray:
        """
        Generates sigma points around $x$ using covariance $P$ in tangent space.

        Parameters
        ----------
        x : np.ndarray
            Current state estimate.
        p_cov : np.ndarray
            Current estimation error covariance (tangent space).

        Returns
        -------
        np.ndarray
            Sigma points array (num_sigmas, dim_x).
        """
        sigmas = [x]

        # Ensure symmetry and stability
        p_sym = (p_cov + p_cov.T) / 2 + np.eye(self.dim_p) * 1e-12

        try:
            l_mat = cholesky((self.dim_p + self.lambda_) * p_sym, lower=True)
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, l_mat[:, i]))
                sigmas.append(self.add_x(x, -l_mat[:, i]))
        except np.linalg.LinAlgError:
            # Fallback for non-PSD matrices
            u_mat = sqrtm((self.dim_p + self.lambda_) * p_sym).real
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, u_mat[i]))
                sigmas.append(self.add_x(x, -u_mat[i]))

        return np.array(sigmas)


class UKF_Attitude(UKF):
    """
    Specialized UKF for Attitude Estimation.
    State: [q0, q1, q2, q3, bias_x, bias_y, bias_z] (7 dim)
    Covariance/Error: 6 dim (tangent space)
    """

    def __init__(self, q_init: np.ndarray | None = None, bias_init: np.ndarray | None = None, dim_z: int = 3, **kwargs: Any) -> None:
        self._quat_mult = quat_mult
        self._quat_conj = quat_conj
        self._axis_angle_to_quat = axis_angle_to_quat
        self._quat_normalize = quat_normalize

        def subtract_x(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            # q1 = q2 * dq => dq = q2_conj * q1
            dq = self._quat_mult(self._quat_conj(x2[:4]), x1[:4])
            if dq[3] < 0:
                dq *= -1
            dtheta = 2 * dq[:3]
            dbias = x1[4:] - x2[4:]
            return np.concatenate([dtheta, dbias])

        def add_x(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
            dq = self._axis_angle_to_quat(dx[:3])
            # q_new = q_old * dq
            q_new = self._quat_normalize(self._quat_mult(x[:4], dq))
            bias_new = x[4:] + dx[3:]
            return np.concatenate([q_new, bias_new])

        def mean_x(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Simple renormalized weighted mean for quaternions
            q_ref = sigmas[0, :4]
            q_avg = np.zeros(4)
            for i in range(len(weights)):
                q = sigmas[i, :4]
                if np.dot(q, q_ref) < 0:
                    q = -q  # Consistent hemisphere
                q_avg += weights[i] * q

            q_avg = self._quat_normalize(q_avg)
            bias_avg = np.dot(weights, sigmas[:, 4:])
            return np.concatenate([q_avg, bias_avg])

        # Default to a small alpha for better local linearity on manifolds
        if "alpha" not in kwargs:
            kwargs["alpha"] = 1e-2

        super().__init__(
            dim_x=7,
            dim_z=dim_z,
            dim_p=6,
            subtract_x=subtract_x,
            add_x=add_x,
            mean_x=mean_x,
            **kwargs,
        )

        if q_init is None:
            q_init = np.array([0, 0, 0, 1.0])
        if bias_init is None:
            bias_init = np.zeros(3)
        self.x = np.concatenate([q_init, bias_init])




