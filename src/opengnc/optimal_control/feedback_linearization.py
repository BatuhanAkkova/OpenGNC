"""
Feedback Linearization Controller for nonlinear systems.
"""

from collections.abc import Callable
from typing import cast

import numpy as np


class FeedbackLinearization:
    r"""
    Feedback Linearization Controller for nonlinear systems.

    Transforms a nonlinear system into an equivalent linear system through
    algebraic state feedback.

    System model: $\dot{x} = f(x) + g(x) u$

    By choosing $u = g(x)^{-1} (v - f(x))$, the closed-loop system becomes:
    $\dot{x} = v$
    where $v$ is the pseudo-control input (linear command).

    Parameters
    ----------
    f_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear drift function $f(x)$.
    g_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear input matrix $g(x)$.
    """

    def __init__(
        self,
        f_func: Callable[[np.ndarray], np.ndarray],
        g_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Initialize the feedback linearization controller."""
        self.f_func = f_func
        self.g_func = g_func

    def compute_control(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute the linearizing control input $u$.

        Parameters
        ----------
        x : np.ndarray
            Current state vector (nx,).
        v : np.ndarray
            Desired linear acceleration/pseudo-control vector (nx,).

        Returns
        -------
        np.ndarray
            The required control input $u$ (nu,).
        """
        x_vec = np.asarray(x)
        v_vec = np.asarray(v)

        f_val = np.asarray(self.f_func(x_vec))
        g_val = np.asarray(self.g_func(x_vec))

        if g_val.size == 1:
            # Scalar case
            return cast(np.ndarray, (v_vec - f_val) / g_val)

        # Matrix case: Solve g(x)u = v - f(x)
        return cast(np.ndarray, np.linalg.solve(g_val, v_vec - f_val))
