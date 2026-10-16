"""
Backstepping Controller for generic 2nd order nonlinear systems.
"""

from collections.abc import Callable
from typing import cast

import numpy as np


class BacksteppingController:
    r"""
    Backstepping Controller for cascaded 2nd-order nonlinear systems.

    Designed for systems of the form:
    $\dot{x}_1 = x_2$
    $\dot{x}_2 = f(x_1, x_2) + g(x_1, x_2) u$

    This implementation uses a standard recursive design to achieve global
    asymptotic stability of a desired trajectory.

    Parameters
    ----------
    f_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Nonlinear drift function $f(x_1, x_2)$.
    g_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Nonlinear input matrix $g(x_1, x_2)$.
    k1 : Union[float, np.ndarray]
        Feedback gain for the first error state (position).
    k2 : Union[float, np.ndarray]
        Feedback gain for the second error state (velocity).
    """

    def __init__(
        self,
        f_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        g_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        k1: float | np.ndarray,
        k2: float | np.ndarray,
    ) -> None:
        """Initialize the backstepping controller parameters."""
        self.f = f_func
        self.g = g_func
        self.k1 = k1
        self.k2 = k2

    def compute_control(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        x1_d: np.ndarray,
        x1_dot_d: np.ndarray,
        x1_ddot_d: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the backstepping control input for the current state.

        Parameters
        ----------
        x1 : np.ndarray
            Measured position/primary state vector (n,).
        x2 : np.ndarray
            Measured velocity/secondary state vector (n,).
        x1_d : np.ndarray
            Desired position/trajectory (n,).
        x1_dot_d : np.ndarray
            Desired velocity (n,).
        x1_ddot_d : np.ndarray, optional
            Desired acceleration (n,). Defaults to zero.

        Returns
        -------
        np.ndarray
            Control input vector $u$ (m,).
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x1_des = np.asarray(x1_d)
        v_des = np.asarray(x1_dot_d)
        a_des = np.asarray(x1_ddot_d) if x1_ddot_d is not None else np.zeros_like(x1_des)

        # 1. Position error (e1)
        e1 = x1 - x1_des

        # 2. Virtual control law (alpha)
        # alpha acts as the 'desired' x2 to stabilize e1
        # alpha = -k1 * e1 + x1_dot_d
        alpha = -(self.k1 @ e1 if isinstance(self.k1, np.ndarray) else self.k1 * e1)
        alpha += v_des

        # 3. Virtual velocity error (e2)
        e2 = x2 - alpha

        # 4. Time derivative of alpha
        # d/dt(alpha) = -k1 * (x2 - x1_dot_d) + x1_ddot_d
        e1_dot = x2 - v_des
        alpha_dot = -(self.k1 @ e1_dot if isinstance(self.k1, np.ndarray) else self.k1 * e1_dot)
        alpha_dot += a_des

        # 5. System dynamics evaluation
        f_val = self.f(x1, x2)
        g_val = self.g(x1, x2)

        # 6. Final Control Law (Step 2)
        # u = g^-1 * (-e1 - f + alpha_dot - k2 * e2)
        inner_term = (
            -e1
            - f_val
            + alpha_dot
            - (self.k2 @ e2 if isinstance(self.k2, np.ndarray) else self.k2 * e2)
        )

        if np.isscalar(g_val) or g_val.shape == () or g_val.shape == (1, 1):
            return cast(np.ndarray, inner_term / g_val)

        # Use pseudoinverse for robust inversion
        return cast(np.ndarray, np.linalg.pinv(g_val) @ inner_term)
