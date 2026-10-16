"""
Control Moment Gyro (CMG) actuator model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.actuators.actuator import Actuator


class ControlMomentGyro(Actuator):
    """
    Control Moment Gyro (CMG) Actuator.

    Models torque produced by changing the orientation of a constant-speed momentum wheel.
    Torque = gimbal_rate x momentum.

    Parameters
    ----------
    wheel_momentum : float
        Angular momentum magnitude (Nms).
    gimbal_axis : np.ndarray
        Unit vector of the gimbal axis (3,).
    spin_axis_init : np.ndarray
        Unit vector of the initial spin axis (3,).
    name : str, optional
        Actuator name. Default is "CMG".
    max_gimbal_rate : float, optional
        Maximum gimbal angular velocity (rad/s).
    """

    def __init__(
        self,
        wheel_momentum: float,
        gimbal_axis: np.ndarray,
        spin_axis_init: np.ndarray,
        name: str = "CMG",
        max_gimbal_rate: float | None = None,
    ) -> None:
        super().__init__(name=name, saturation=max_gimbal_rate)
        self.h_mag = wheel_momentum
        self.g_axis = np.array(gimbal_axis) / np.linalg.norm(gimbal_axis)
        self.s_axis = np.array(spin_axis_init) / np.linalg.norm(spin_axis_init)

        # Verify orthogonality (approximately)
        if abs(np.dot(self.g_axis, self.s_axis)) > 1e-6:
            # Re-orthogonalize s_axis
            self.s_axis = self.s_axis - np.dot(self.s_axis, self.g_axis) * self.g_axis
            self.s_axis /= np.linalg.norm(self.s_axis)

        self.t_axis = np.cross(self.g_axis, self.s_axis)  # Transverse axis
        self.gimbal_angle = 0.0

    def get_axes(self, angle: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current spin and transverse axes for a given gimbal angle.

        Parameters
        ----------
        angle : float, optional
            Gimbal angle (rad). If None, uses current `self.gimbal_angle`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spin_axis, transverse_axis).
        """
        theta = angle if angle is not None else self.gimbal_angle
        # Rotation about gimbal axis
        s = self.s_axis * np.cos(theta) + self.t_axis * np.sin(theta)
        t = np.cross(self.g_axis, s)
        return s, t

    def command(
        self, gimbal_rate_cmd: Any = None, dt: float | None = None, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate torque produced by gimbal rate.

        Parameters
        ----------
        gimbal_rate_cmd : float
            Commanded gimbal rate (rad/s).
        dt : float, optional
            Time step (s) to update gimbal angle.
        **kwargs : dict
            Additional parameters (unused).

        Returns
        -------
        np.ndarray
            Torque vector (Nm) (3,).
        """
        # Apply rate limits
        if gimbal_rate_cmd is None:
            if not args:
                raise ValueError("gimbal_rate_cmd is required.")
            gimbal_rate_cmd = args[0]
        g_rate = float(self.apply_saturation(float(gimbal_rate_cmd)))

        # Current axes
        _, t = self.get_axes()

        # Torque = g_rate * h_mag * (g x s) = g_rate * h_mag * t
        torque_vec = g_rate * self.h_mag * t

        # Update gimbal angle
        if dt is not None:
            self.gimbal_angle += g_rate * dt
            # Normalize angle to [-pi, pi]
            self.gimbal_angle = (self.gimbal_angle + np.pi) % (2 * np.pi) - np.pi

        return torque_vec

    def get_torque_jacobian(self, angle: float | None = None) -> np.ndarray:
        """
        Returns the Jacobian mapping gimbal rate to torque: T = A * g_rate.

        Parameters
        ----------
        angle : float, optional
            Angle to evaluate at (rad).

        Returns
        -------
        np.ndarray
            (3, 1) mapping matrix.
        """
        _, t = self.get_axes(angle)
        return (self.h_mag * t).reshape(3, 1)
