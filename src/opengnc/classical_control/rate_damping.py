"""
Proportional rate damping controller for torque-based detumbling.
"""

from typing import cast

import numpy as np


class RateDampingControl:
    r"""
    Proportional angular rate damping controller for spacecraft detumbling.

    Generates torque commands to reduce the spacecraft's angular rates,
    typically using thrusters or other active actuators.

    Control law: ``T = -K omega``.
    """

    def __init__(self, gain: float, max_torque: float | None = None) -> None:
        """Initialize the rate damping controller."""
        self.gain = gain
        self.max_torque = max_torque

    def compute_torque(self, omega: np.ndarray) -> np.ndarray:
        """Compute the commanded damping torque."""
        torque_raw = -self.gain * np.asarray(omega, dtype=float)

        if self.max_torque is not None:
            norm_t = np.linalg.norm(torque_raw)
            if norm_t > self.max_torque:
                return cast(
                    np.ndarray, np.asarray(torque_raw * (self.max_torque / norm_t), dtype=float)
                )

        return cast(np.ndarray, np.asarray(torque_raw, dtype=float))
