"""
Linear Quadratic Regulator (LQR) Controller.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


class LQR:
    r"""
    Linear Quadratic Regulator (LQR).

    Minimizes:
    $J = \int_0^\infty (\mathbf{x}^T \mathbf{Q} \mathbf{x} + \mathbf{u}^T \mathbf{R} \mathbf{u}) dt$

    Optimal Law: $\mathbf{u} = -\mathbf{K} \mathbf{x}$

    Parameters
    ----------
    A : np.ndarray
        State matrix ($n \times n$).
    B : np.ndarray
        Input matrix ($n \times m$).
    Q : np.ndarray
        Weight matrix ($n \times n$, PSD).
    R : np.ndarray
        Weight matrix ($m \times m$, PD).
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
        """Initialize the LQR controller with system and cost matrices."""
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.P: np.ndarray | None = None
        self.K: np.ndarray | None = None

    def solve(self) -> np.ndarray:
        """
        Solve the Continuous Algebraic Riccati Equation (CARE).

        $A^T P + P A - P B R^{-1} B^T P + Q = 0$

        Returns
        -------
        np.ndarray
            The unique positive-definite solution matrix P.
        """
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        return self.P

    def compute_gain(self) -> np.ndarray:
        """
        Compute the optimal feedback gain matrix K.

        $K = R^{-1} B^T P$

        Returns
        -------
        np.ndarray
            Feedback gain matrix K (m x n).
        """
        if self.P is None:
            self.solve()
        if self.P is None:
            raise RuntimeError("Riccati solution not available.")

        # Numerically stable solve for R*K = B^T*P
        self.K = np.linalg.solve(self.R, self.B.T @ self.P)
        return self.K
