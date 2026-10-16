"""
Linear Quadratic Gaussian (LQG) Controller.
"""

import numpy as np

from .lqe import LQE
from .lqr import LQR


class LQG:
    r"""
    Linear Quadratic Gaussian (LQG) Controller.

    Integrates LQR optimal state-feedback with a Linear Quadratic Estimator
    (LQE/Kalman Filter). Operates on the 'Separation Principle', where the
    controller and observer are designed independently.

    System Model:
    $\dot{x} = Ax + Bu + Gw$
    $y = Cx + v$

    Control Law: $u = -K \hat{x}$
    Observer: $\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$

    Parameters
    ----------
    A : np.ndarray
        State matrix (nx x nx).
    B : np.ndarray
        Input matrix (nx x nu).
    C : np.ndarray
        Output matrix (ny x nx).
    Q_lqr : np.ndarray
        State cost matrix for LQR.
    R_lqr : np.ndarray
        Input cost matrix for LQR.
    Q_lqe : np.ndarray
        Process noise covariance matrix for estimator.
    R_lqe : np.ndarray
        Measurement noise covariance matrix for estimator.
    G_lqe : np.ndarray, optional
        Process noise input matrix. Defaults to Identity (nx x nx).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q_lqr: np.ndarray,
        R_lqr: np.ndarray,
        Q_lqe: np.ndarray,
        R_lqe: np.ndarray,
        G_lqe: np.ndarray | None = None,
    ) -> None:
        """Initialize the LQG controller and compute LQR/LQE gains."""
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)

        self.nx = self.A.shape[0]
        if G_lqe is None:
            G_lqe = np.eye(self.nx)

        # Design LQR optimal gain
        self.lqr = LQR(A, B, Q_lqr, R_lqr)
        self.K = self.lqr.compute_gain()

        # Design LQE optimal observer gain
        self.lqe = LQE(A, G_lqe, C, Q_lqe, R_lqe)
        self.L = self.lqe.compute_gain()

        # Initialize state estimate to zero
        self.x_hat = np.zeros(self.nx)

    def update_estimation(self, y: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        r"""
        Update the internal state estimate using observer dynamics.

        Parameters
        ----------
        y : np.ndarray
            Latest sensor measurement vector (ny,).
        u : np.ndarray
            The control input vector actually applied (nu,).
        dt : float
            Time step since the last update (s).

        Returns
        -------
        np.ndarray
            The updated state estimate $\hat{x}$ (nx,).
        """
        # Observer update using Euler integration
        innovation = np.asarray(y) - (self.C @ self.x_hat)
        x_hat_dot = (self.A @ self.x_hat) + (self.B @ np.asarray(u)) + (self.L @ innovation)
        self.x_hat += x_hat_dot * dt
        return self.x_hat

    def compute_control(
        self,
        y: np.ndarray | None = None,
        dt: float | None = None,
        u_last: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Compute the optimal control input based on the state estimate.

        Parameters
        ----------
        y : np.ndarray, optional
            New measurement for estimate update.
        dt : float, optional
            Time step for estimate update.
        u_last : np.ndarray, optional
            Previous control input for estimate update.

        Returns
        -------
        np.ndarray
            Optimal control input $u = -K\hat{x}$ (nu,).
        """
        if y is not None and dt is not None:
            # Predict then update estimate
            u_applied = u_last if u_last is not None else -self.K @ self.x_hat
            self.update_estimation(y, u_applied, dt)

        return -self.K @ self.x_hat
