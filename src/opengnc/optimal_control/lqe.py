"""
Linear Quadratic Estimator (LQE) / Kalman Filter.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class LQE:
    r"""
    Linear Quadratic Estimator (LQE) / Kalman Filter gain designer.

    Computes the optimal steady-state observer gain $L$ for a continuous-time
    system with additive Gaussian process and measurement noise.

    System Model:
    $\dot{x} = Ax + Bu + Gw$
    $y = Cx + v$
    where $Q = E[ww^T]$ and $R = E[vv^T]$.

    The resulting observer dynamics are:
    $\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$

    Parameters
    ----------
    A : np.ndarray
        State matrix (nx x nx).
    G : np.ndarray
        Process noise input matrix (nx x nw). Often Identity.
    C : np.ndarray
        Output matrix (ny x nx).
    Q : np.ndarray
        Process noise covariance matrix (nw x nw).
    R : np.ndarray
        Measurement noise covariance matrix (ny x ny).

    Attributes
    ----------
    P : np.ndarray
        Steady-state estimation error covariance matrix.
    L : np.ndarray
        Optimal observer gain matrix (nx x ny).
    """

    def __init__(
        self,
        A: np.ndarray,
        G: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """Initialize the LQE parameters."""
        self.A = np.asarray(A)
        self.G = np.asarray(G)
        self.C = np.asarray(C)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.P: np.ndarray | None = None
        self.L: np.ndarray | None = None

    def solve(self) -> np.ndarray:
        """
        Solve the Estimation Algebraic Riccati Equation (ARE).

        $A P + P A^T - P C^T R^{-1} C P + G Q G^T = 0$

        Returns
        -------
        np.ndarray
            The unique positive-definite solution matrix P.
        """
        # Mapping to solve_continuous_are:
        # A_eff = A.T, B_eff = C.T, Q_eff = G*Q*G.T, R_eff = R
        a_eff = self.A.T
        b_eff = self.C.T
        q_eff = self.G @ self.Q @ self.G.T
        r_eff = self.R

        self.P = solve_continuous_are(a_eff, b_eff, q_eff, r_eff)
        return self.P

    def compute_gain(self) -> np.ndarray:
        """
        Compute the optimal observer gain matrix L.

        $L = P C^T R^{-1}$

        Returns
        -------
        np.ndarray
            Observer gain matrix L (nx x ny).
        """
        if self.P is None:
            self.solve()
        if self.P is None:
            raise RuntimeError("Riccati solution not available.")

        # Numerically stable solve for R*L.T = C*P
        term = np.linalg.solve(self.R, self.C @ self.P)
        self.L = term.T

        return self.L
