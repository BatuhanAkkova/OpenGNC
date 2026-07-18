"""
Passivity-Based Controller for Euler-Lagrange mechanical systems.
"""

from collections.abc import Callable
from typing import cast

import numpy as np


class PassivityBasedController:
    r"""
    Passivity-Based Controller for Euler-Lagrange mechanical systems.

    Exploits the energy and passivity properties of mechanical systems to
    ensure global stability. This implementation is based on the Slotine & Li
    adaptive/passivity scheme.

    System model: $M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = u$

    Parameters
    ----------
    M_func : Callable[[np.ndarray], np.ndarray]
        Inertia matrix function $M(q)$ (n x n).
    C_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Coriolis and centrifugal matrix function $C(q, \dot{q})$ (n x n).
    G_func : Callable[[np.ndarray], np.ndarray]
        Gravity/damping vector function $G(q)$ (n,).
    K_d : Union[float, np.ndarray]
        Dissipative (damping) gain matrix (n x n).
    Lambda : Union[float, np.ndarray]
        Proportional error convergence matrix (n x n).
    """

    def __init__(
        self,
        M_func: Callable[[np.ndarray], np.ndarray],
        C_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        G_func: Callable[[np.ndarray], np.ndarray],
        K_d: float | np.ndarray,
        Lambda: float | np.ndarray,
    ) -> None:
        """Initialize controller gains and model functions."""
        self.M = M_func
        self.C = C_func
        self.G = G_func
        self.K_d = K_d
        self.Lambda = Lambda

    def compute_control(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_d: np.ndarray,
        q_dot_d: np.ndarray,
        q_ddot_d: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the passivity-based control torque output.

        Parameters
        ----------
        q, q_dot : np.ndarray
            Current generalized coordinates and velocities (n,).
        q_d, q_dot_d : np.ndarray
            Desired coordinate and velocity trajectories (n,).
        q_ddot_d : np.ndarray, optional
            Desired feedforward acceleration (n,). Defaults to zero.

        Returns
        -------
        np.ndarray
            Control input vector $u$ (n,).
        """
        q_vec = np.asarray(q)
        v_vec = np.asarray(q_dot)
        qd_vec = np.asarray(q_d)
        vd_vec = np.asarray(q_dot_d)
        ad_vec = np.asarray(q_ddot_d) if q_ddot_d is not None else np.zeros_like(qd_vec)

        # 1. Coordinate errors
        e_q = q_vec - qd_vec
        e_v = v_vec - vd_vec

        # 2. Reference velocity trajectory (v_r)
        # v_r defines the manifold for asymptotic convergence
        # v_r = q_dot_d - Lambda * e_q
        v_r = vd_vec - (self.Lambda @ e_q if isinstance(self.Lambda, np.ndarray) else self.Lambda * e_q)

        # 3. Reference acceleration trajectory (v_r_dot)
        v_r_dot = ad_vec - (self.Lambda @ e_v if isinstance(self.Lambda, np.ndarray) else self.Lambda * e_v)

        # 4. Tracking surface / sliding error (s)
        # s = q_dot - v_r = e_dot + Lambda * e
        s = v_vec - v_r

        # 5. Evaluate System Matrices at current configuration
        M_mat = self.M(q_vec)
        C_mat = self.C(q_vec, v_vec)
        G_vec = self.G(q_vec)

        # 6. Control Law: Feedforward + Damping
        # u = M*v_r_dot + C*v_r + G - K_d*s
        ff_term = M_mat @ v_r_dot + C_mat @ v_r + G_vec
        damp_term = self.K_d @ s if isinstance(self.K_d, np.ndarray) else self.K_d * s

        return cast(np.ndarray, ff_term - damp_term)




