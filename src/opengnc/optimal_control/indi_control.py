from collections.abc import Callable
from typing import cast

import numpy as np


class INDIController:
    r"""
    Incremental Nonlinear Dynamic Inversion (INDI) Controller.

    A sensor-based control method that reduces model dependency by
    calculating control increments based on measured accelerations.

    System model: $\ddot{x} = f(x, \dot{x}) + g(x, \dot{x}) u$
    Discrete Law: $u_k = u_{k-1} + g(x_k, \dot{x}_k)^{-1} (v_k - \ddot{x}_k)$

    where $\ddot{x}_k$ is the current measured acceleration and $v_k$ is the
    desired acceleration command.

    Parameters
    ----------
    g_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Nonlinear input matrix function $g(x, \dot{x})$.
    """

    def __init__(self, g_func: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """Initialize INDI controller with input matrix function."""
        self.g = g_func

    def compute_control(
        self,
        u0: np.ndarray,
        x_ddot0: np.ndarray,
        v: np.ndarray,
        x0: np.ndarray,
        x_dot0: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the incremental control input $u$.

        Parameters
        ----------
        u0 : np.ndarray
            Previous control input vector (nu,).
        x_ddot0 : np.ndarray
            Current measured/estimated acceleration vector (nx,).
        v : np.ndarray
            Desired acceleration pseudo-control vector (nx,).
        x0 : np.ndarray
            Current state/position vector (nx,).
        x_dot0 : np.ndarray
            Current velocity vector (nx,).

        Returns
        -------
        np.ndarray
            Optimal control input $u$ (nu,).
        """
        u_prev = np.asarray(u0)
        acc_meas = np.asarray(x_ddot0)
        v_cmd = np.asarray(v)
        x_curr = np.asarray(x0)
        v_curr = np.asarray(x_dot0)

        g_val = self.g(x_curr, v_curr)

        # Incremental law: delta_u = g^-1 (v - x_ddot_measured)
        acc_err = v_cmd - acc_meas

        if np.isscalar(g_val) or g_val.shape == () or g_val.shape == (1, 1):
            delta_u = acc_err / g_val
        else:
            # Use pseudo-inverse for robust increments
            delta_u = np.linalg.pinv(g_val) @ acc_err

        return cast(np.ndarray, u_prev + delta_u)


class INDIOuterLoopPD:
    """
    Standard PD Outer-Loop for INDI acceleration command generation.

    Parameters
    ----------
    Kp : Union[float, np.ndarray]
        Proportional gain matrix or scalar.
    Kd : Union[float, np.ndarray]
        Derivative gain matrix or scalar.
    """

    def __init__(self, Kp: float | np.ndarray, Kd: float | np.ndarray) -> None:
        """Initialize outer-loop tracking gains."""
        self.Kp = Kp
        self.Kd = Kd

    def compute_v(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        x_d: np.ndarray,
        x_dot_d: np.ndarray,
        x_ddot_d: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute desired acceleration pseudo-control $v$.

        Parameters
        ----------
        x, x_dot : np.ndarray
            Current position and velocity measurements.
        x_d, x_dot_d : np.ndarray
            Desired position and velocity trajectories.
        x_ddot_d : np.ndarray, optional
            Desired feedforward acceleration. Defaults to zero.

        Returns
        -------
        np.ndarray
            Acceleration command $v$.
        """
        x_val = np.asarray(x)
        v_val = np.asarray(x_dot)
        xd_val = np.asarray(x_d)
        vd_val = np.asarray(x_dot_d)
        ad_val = np.asarray(x_ddot_d) if x_ddot_d is not None else np.zeros_like(xd_val)

        err_p = x_val - xd_val
        err_d = v_val - vd_val

        p_term = self.Kp @ err_p if isinstance(self.Kp, np.ndarray) else self.Kp * err_p
        d_term = self.Kd @ err_d if isinstance(self.Kd, np.ndarray) else self.Kd * err_d

        return cast(np.ndarray, ad_val - p_term - d_term)




