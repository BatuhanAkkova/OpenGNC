"""
H2 Optimal Controller (LQG equivalent).
"""

import numpy as np

from .lqg import LQG


class H2Controller(LQG):
    """
    H2-Optimal Controller.

    Solves the standard H2 control problem for linear state-space systems.
    For systems with Gaussian noise and quadratic performance indices, the
    H2-optimal controller is equivalent to the Linear Quadratic Gaussian (LQG)
    controller.

    Parameters
    ----------
    A : np.ndarray
        State matrix (nx x nx).
    B : np.ndarray
        Control input matrix (nx x nu).
    C : np.ndarray
        Output matrix (ny x nx).
    Q_lqr : np.ndarray
        State weighting matrix for LQR (nx x nx).
    R_lqr : np.ndarray
        Control weighting matrix for LQR (nu x nu).
    Q_lqe : np.ndarray
        Process noise covariance matrix for LQE (nw x nw).
    R_lqe : np.ndarray
        Measurement noise covariance matrix for LQE (ny x ny).
    G_lqe : np.ndarray, optional
        Process noise input matrix (nx x nw). Defaults to Identity.
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
        """Initialize H2 controller as an LQG instance."""
        super().__init__(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe, G_lqe)

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve LQR and LQE optimal gain design problems.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Optimal feedback gain K (nu x nx) and observer gain L (nx x ny).
        """
        self.lqr.solve()
        self.lqe.solve()
        self.K = self.lqr.compute_gain()
        self.L = self.lqe.compute_gain()
        return self.K, self.L
