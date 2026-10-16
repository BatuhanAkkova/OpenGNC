"""
Sliding Mode Controller (SMC) implementation.
"""

from collections.abc import Callable
from typing import cast

import numpy as np


class SlidingModeController:
    r"""
    Sliding Mode Controller (SMC).

    Control Law:
    $u = u_{eq} - K \cdot \text{sat}(s/\Phi)$

    Parameters
    ----------
    surface_func : Callable
        Sliding surface $s(\mathbf{x}, t)$.
    k_gain : float
        Switching gain $K$.
    equivalent_control_func : Callable | None, optional
        $\mathbf{u}_{eq}$ component.
    chattering_reduction : bool, optional
        Use saturation instead of sign. Default True.
    boundary_layer : float, optional
        Saturation boundary $\Phi$.
    """

    def __init__(
        self,
        surface_func: Callable[[np.ndarray, float], float],
        k_gain: float,
        equivalent_control_func: Callable[[np.ndarray, float], float] | None = None,
        chattering_reduction: bool = True,
        boundary_layer: float = 0.1,
    ) -> None:
        """Initialize the Sliding Mode Controller."""
        self.surface_func = surface_func
        self.k_gain = k_gain
        self.eq_func = equivalent_control_func if equivalent_control_func else lambda x, t: 0.0
        self.use_sat = chattering_reduction
        self.phi = boundary_layer

    def compute_control(self, x: np.ndarray, t: float = 0.0) -> float | np.ndarray:
        """
        Compute the sliding mode control input.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        t : float, optional
            Current time (s). Default is 0.0.

        Returns
        -------
        float or np.ndarray
            The computed control input signal.
        """
        s_val = self.surface_func(x, t)

        if self.use_sat:
            # Saturation function: sat(s/phi) used for chattering reduction
            switching_term = np.clip(s_val / self.phi, -1.0, 1.0)
        else:
            switching_term = np.sign(s_val)

        return cast(float | np.ndarray, self.eq_func(x, t) - self.k_gain * switching_term)
