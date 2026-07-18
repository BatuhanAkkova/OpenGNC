"""
Extended Kalman Filter for Orbit Determination (OD-EKF).
"""

from typing import Any, cast

import numpy as np

from opengnc.disturbances.gravity import J2Gravity, TwoBodyGravity
from opengnc.kalman_filters.ekf import EKF


class OrbitDeterminationEKF:
    r"""
    Extended Kalman Filter for Orbit Determination (OD-EKF).

    Estimates the 6D Cartesian state $\mathbf{x} = [\mathbf{r}, \mathbf{v}]^T$
    in the ECI frame. Supports multi-model dynamics (Two-body, J2) and
    RK4-based state prediction.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector $[r_x, r_y, r_z, v_x, v_y, v_z]$ (m, m/s).
    p0 : np.ndarray
        Initial $6\times 6$ estimation error covariance matrix.
    q_mat : np.ndarray
        Process noise covariance matrix $\mathbf{Q} \in \mathbb{R}^{6\times 6}$.
    r_mat : np.ndarray
        Measurement noise covariance matrix $\mathbf{R} \in \mathbb{R}^{3\times 3}$.
    use_j2 : bool, optional
        Whether to include J2 perturbations. Default is True.
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    re : float, optional
        Earth radius (m).
    """

    def __init__(
        self,
        x0: np.ndarray,
        p0: np.ndarray,
        q_mat: np.ndarray,
        r_mat: np.ndarray,
        use_j2: bool = True,
        mu: float = 398600.4418e9,
        re: float = 6378137.0,
    ) -> None:
        r"""
        Initialize the Orbit Determination EKF.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state $[r_x, r_y, r_z, v_x, v_y, v_z]$ (m, m/s).
        p0 : np.ndarray
            Initial covariance $6 \times 6$.
        q_mat : np.ndarray
            Process noise covariance.
        r_mat : np.ndarray
            Measurement noise covariance.
        use_j2 : bool, optional
            Include J2 gravity perturbations. Default True.
        mu : float, optional
            Gravitational parameter ($m^3/s^2$).
        re : float, optional
            Planet radius (m).
        """
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = cast(np.ndarray, np.asarray(x0, dtype=float))
        self.ekf.P = cast(np.ndarray, np.asarray(p0, dtype=float))
        self.ekf.Q = cast(np.ndarray, np.asarray(q_mat, dtype=float))
        self.ekf.R = cast(np.ndarray, np.asarray(r_mat, dtype=float))

        self.mu = mu
        self.re = re
        self.use_j2 = use_j2

        self.gravity: TwoBodyGravity | J2Gravity
        if use_j2:
            self.gravity = J2Gravity(mu=mu, re=re)
        else:
            self.gravity = TwoBodyGravity(mu=mu)

    def _state_transition(
        self,
        x: np.ndarray,
        dt: float,
        u: np.ndarray | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        r"""
        Integrate orbital dynamics using the RK4 method.

        Parameters
        ----------
        x : np.ndarray
            Current state.
        dt : float
            Integration step (s).
        u : np.ndarray, optional
            Control input. Unused.

        Returns
        -------
        np.ndarray
            Propagated state $\mathbf{x}(t + \Delta t)$.
        """
        def f_dynamics(state: np.ndarray) -> np.ndarray:
            r = state[:3]
            v = state[3:]
            a = self.gravity.get_acceleration(r)
            return np.concatenate([v, a])

        k1 = f_dynamics(x)
        k2 = f_dynamics(x + 0.5 * dt * k1)
        k3 = f_dynamics(x + 0.5 * dt * k2)
        k4 = f_dynamics(x + dt * k3)

        return cast(np.ndarray, x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    def _jacobian_f(
        self,
        x: np.ndarray,
        dt: float,
        u: np.ndarray | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        r"""
        Compute the Discrete-Time State Transition Jacobian.

        Linearization uses the Gravity Gradient Matrix:
        $\mathbf{G}(\mathbf{r}) = \frac{\mu}{r^3} \left[ \frac{3 \mathbf{r} \mathbf{r}^T}{r^2} - \mathbf{I} \right]$

        Parameters
        ----------
        x : np.ndarray
            Current state.
        dt : float
            Time step (s).
        u : np.ndarray | None, optional
            Control input.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        np.ndarray
            $6 \times 6$ transition matrix $\mathbf{F} \approx \mathbf{I} + \mathbf{A} \Delta t$.
        """
        r_vec = x[:3]
        r_mag = np.linalg.norm(r_vec)

        # Continuous-time Gravity Gradient Matrix
        g_mat = (self.mu / r_mag**3) * (3.0 * np.outer(r_vec, r_vec) / r_mag**2 - np.eye(3))

        a_mat = np.zeros((6, 6))
        a_mat[:3, 3:] = np.eye(3)
        a_mat[3:, :3] = g_mat

        return cast(np.ndarray, np.eye(6) + a_mat * dt)

    def predict(self, dt: float) -> None:
        """
        Perform EKF prediction step ($x^-, P^-$).

        Parameters
        ----------
        dt : float
            Step size (s).
        """
        self.ekf.predict(self._state_transition, self._jacobian_f, dt)

    def update(self, z_pos: np.ndarray) -> None:
        r"""
        Update state estimate using position measurement $\mathbf{z} = \mathbf{r} + \nu$.

        Parameters
        ----------
        z_pos : np.ndarray
            ECI Position measurement (m).
        """
        def hx(x_state: np.ndarray) -> np.ndarray:
            return x_state[:3]

        def h_jac(x_state_unused: np.ndarray) -> np.ndarray:
            h_mat = np.zeros((3, 6))
            h_mat[:, :3] = np.eye(3)
            return h_mat

        self.ekf.update(np.asarray(z_pos), hx, h_jac)

    @property
    def state(self) -> np.ndarray:
        """Estimated orbit state vector $[r, v]^T$."""
        return self.ekf.x

    @property
    def covariance(self) -> np.ndarray:
        r"""Estimation error covariance matrix $\mathbf{P}$."""
        return self.ekf.P




