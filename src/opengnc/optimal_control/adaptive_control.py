"""
Model Reference Adaptive Control (MRAC) for state-space systems.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class ModelReferenceAdaptiveControl:
    r"""
    Model Reference Adaptive Controller (MRAC) for linear state-space systems.

    Implements a direct MRAC scheme where controller parameters are updated to
    minimize the error between the plant and a stable reference model.

    Plant with parametric uncertainty:
    $\dot{x} = Ax + B(u + \Theta^T \Phi(x))$

    Reference Model:
    $\dot{x}_m = A_m x_m + B_m r$

    Adaptation Law (Lyapunov-based):
    $\dot{\hat{\Theta}} = \Gamma \Phi(x) e^T P B$
    where $e = x - x_m$, and $P$ solves $A_m^T P + P A_m = -Q$.

    Parameters
    ----------
    A_m : np.ndarray
        Reference model state matrix (nx x nx).
    B_m : np.ndarray
        Reference model input matrix (nx x nu).
    B : np.ndarray
        Plant input matrix (nx x nu).
    Gamma : np.ndarray
        Adaptation gain matrix (nk x nk).
    Q_lyap : np.ndarray
        Positive-definite matrix for the Lyapunov equation (nx x nx).
    phi_func : Callable[[np.ndarray], np.ndarray]
        Regressor function $\Phi(x)$ returning a vector of basis functions (nk,).
    """

    def __init__(
        self,
        A_m: np.ndarray,
        B_m: np.ndarray,
        B: np.ndarray,
        Gamma: np.ndarray,
        Q_lyap: np.ndarray,
        phi_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Initialize the MRAC parameters and solve the Lyapunov equation."""
        self.A_m = np.asarray(A_m)
        self.B_m = np.asarray(B_m)
        self.B = np.asarray(B)
        self.Gamma = np.asarray(Gamma)
        self.phi = phi_func

        # Solve for P in A_m.T * P + P * A_m = -Q
        self.P = solve_continuous_lyapunov(self.A_m.T, -np.asarray(Q_lyap))

        self.nx = self.A_m.shape[0]
        self.nu = self.B_m.shape[1]

        # Determine number of parameters from phi
        dummy_x = np.zeros(self.nx)
        dummy_phi = self.phi(dummy_x)
        self.nk = len(dummy_phi)

        # Parameter estimate \hat{\Theta} (nk x nu)
        self.theta_hat = np.zeros((self.nk, self.nu))
        self.d_theta_hat = np.zeros_like(self.theta_hat)

    def compute_control(
        self,
        x: np.ndarray,
        x_m: np.ndarray,
        r: np.ndarray,
        kx: np.ndarray | None = None,
        kr: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the adaptive control input and the parameter update rate.

        Parameters
        ----------
        x : np.ndarray
            Current plant state vector (nx,).
        x_m : np.ndarray
            Current reference model state vector (nx,).
        r : np.ndarray
            Reference command input (nu,).
        kx : np.ndarray, optional
            Nominal feedback gain s.t. A + B*Kx = Am. Default is zero.
        kr : np.ndarray, optional
            Nominal feedforward gain s.t. B*Kr = Bm. Default is Identity.

        Returns
        -------
        np.ndarray
            Control input vector $u$ (nu,).
        """
        x_vec = np.asarray(x)
        xm_vec = np.asarray(x_m)
        r_vec = np.asarray(r).flatten()

        # Defaults for matching gains
        k_x = kx if kx is not None else np.zeros((self.nu, self.nx))
        k_r = kr if kr is not None else np.eye(self.nu)

        phi_x = self.phi(x_vec).reshape(-1, 1)  # (nk, 1)
        adaptive_term = (self.theta_hat.T @ phi_x).flatten()

        u = (k_x @ x_vec) + (k_r @ r_vec) - adaptive_term

        # Compute adaptation rate: Gamma * Phi(x) * e^T * P * B
        e = (x_vec - xm_vec).reshape(1, -1)
        self.d_theta_hat = self.Gamma @ phi_x @ (e @ self.P @ self.B)

        return cast(np.ndarray, u)

    def update_theta(self, dt: float) -> None:
        """
        Integrate the parameter estimates using the computed update rate.

        Should be called once per simulation step after compute_control.

        Parameters
        ----------
        dt : float
            Simulation time step (s).
        """
        self.theta_hat += self.d_theta_hat * dt
