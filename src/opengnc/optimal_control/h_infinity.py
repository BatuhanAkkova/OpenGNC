"""
H-Infinity Robust Controller state-feedback design.
"""

from typing import cast

import numpy as np
from scipy.linalg import solve_continuous_are


class HInfinityController:
    r"""
    H-Infinity Robust State-Feedback Controller.

    Designs a control law $u = -Kx$ that minimizes the $H_\infty$ norm of the
    closed-loop transfer function from external disturbances $w$ to performance
    outputs $z$, ensuring robustness against worst-case disturbances.

    System Dynamics: $\dot{x} = Ax + B_1 w + B_2 u$
    Cost Function: $J = \int_{0}^{\infty} (x^T Q x + u^T R u - \gamma^2 w^T w) dt$

    Parameters
    ----------
    A : np.ndarray
        State matrix (nx x nx).
    B1 : np.ndarray
        Disturbance input matrix (nx x nw).
    B2 : np.ndarray
        Control input matrix (nx x nu).
    Q : np.ndarray
        State cost matrix (nx x nx).
    R : np.ndarray
        Control cost matrix (nu x nu).
    gamma : float
        Target $L_2$-gain attenuation level (robustness margin).
    """

    def __init__(
        self,
        A: np.ndarray,
        B1: np.ndarray,
        B2: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        gamma: float,
    ) -> None:
        """Initialize the H-infinity controller parameters."""
        self.A = np.asarray(A)
        self.B1 = np.asarray(B1)
        self.B2 = np.asarray(B2)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.gamma = float(gamma)
        self.P: np.ndarray | None = None
        self.K: np.ndarray | None = None

    def solve(self) -> np.ndarray:
        r"""
        Solve the H-infinity Algebraic Riccati Equation (ARE).

        $A^T P + P A + P ( \frac{1}{\gamma^2} B_1 B_1^T - B_2 R^{-1} B_2^T ) P + Q = 0$

        Returns
        -------
        np.ndarray
            The positive-definite Riccati solution matrix P.

        Raises
        ------
        ValueError
            If no solution exists for the given attenuation level gamma.
        """
        # Effective control-minus-disturbance weighting
        # S = B2*inv(R)*B2.T - (1/gamma^2)*B1*B1.T
        r_inv = np.linalg.inv(self.R)
        s_mat = self.B2 @ r_inv @ self.B2.T - (1.0 / self.gamma**2) * (self.B1 @ self.B1.T)

        try:
            # We use use solve_continuous_are which expects: A.T*P + P*A - P*B*R^-1*B.T*P + Q = 0
            # So we pass B s.t. B*R^-1*B.T = S
            self.P = solve_continuous_are(self.A, np.eye(self.A.shape[0]), self.Q, np.linalg.inv(s_mat))
            return self.P
        except Exception as e:
            raise ValueError(
                f"H-infinity solution does not exist for gamma={self.gamma}. Try increasing gamma. Error: {e}"
            )

    def compute_gain(self) -> np.ndarray:
        """
        Compute the optimal robust feedback gain matrix K.

        $K = R^{-1} B_2^T P$

        Returns
        -------
        np.ndarray
            Feedback gain matrix K (nu x nx).
        """
        if self.P is None:
            self.solve()
        if self.P is None:
            raise RuntimeError("Riccati solution not available.")

        self.K = np.linalg.solve(self.R, self.B2.T @ self.P)
        return self.K

    def compute_control(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the control input based on the state vector.

        Parameters
        ----------
        x : np.ndarray
            Current state vector (nx,).

        Returns
        -------
        np.ndarray
            Control input vector $u = -Kx$ (nu,).
        """
        if self.K is None:
            self.compute_gain()
        if self.K is None:
            raise RuntimeError("Feedback gain not available.")
        return cast(np.ndarray, -self.K @ x)




