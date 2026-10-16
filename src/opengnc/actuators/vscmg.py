"""
Variable Speed Control Moment Gyro (VSCMG) actuator model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.actuators.cmg import ControlMomentGyro


class VariableSpeedCMG(ControlMomentGyro):
    """
    Variable Speed Control Moment Gyro (VSCMG).

    Allows wheel speed to change (Reaction Wheel effect) while gimbaling (CMG effect).
    Torque = I_w * alpha * s + gimbal_rate * h * t.

    Parameters
    ----------
    wheel_inertia : float
        Moment of inertia of the wheel (kg*m^2).
    gimbal_axis : np.ndarray
        Gimbal axis vector (3,).
    spin_axis_init : np.ndarray
        Initial spin axis vector (3,).
    name : str, optional
        Actuator name. Default is "VSCMG".
    max_gimbal_rate : float, optional
        Maximum gimbal angular velocity (rad/s).
    max_wheel_torque : float, optional
        Maximum torque for wheel acceleration (Nm).
    """

    def __init__(
        self,
        wheel_inertia: float,
        gimbal_axis: np.ndarray,
        spin_axis_init: np.ndarray,
        name: str = "VSCMG",
        max_gimbal_rate: float | None = None,
        max_wheel_torque: float | None = None,
    ) -> None:
        super().__init__(
            wheel_momentum=0.0,
            gimbal_axis=gimbal_axis,
            spin_axis_init=spin_axis_init,
            name=name,
            max_gimbal_rate=max_gimbal_rate,
        )

        self.inertia = wheel_inertia
        self.wheel_speed = 0.0
        self.max_wheel_torque = max_wheel_torque

    def command(
        self,
        gimbal_rate_cmd: Any = None,
        dt: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Calculate combined torque from gimbaling and wheel acceleration.

        Parameters
        ----------
        cmd_vec : tuple[float, float] | list[float]
            (gimbal_rate_cmd, wheel_torque_cmd).
            - gimbal_rate_cmd : Commanded gimbal rate (rad/s).
            - wheel_torque_cmd : Commanded wheel torque (Nm).
        dt : float, optional
            Time step (s) for state integration.

        Returns
        -------
        np.ndarray
            Net torque vector (Nm) (3,).
        """
        cmd_vec = kwargs.get("cmd_vec", None)
        if cmd_vec is None:
            if gimbal_rate_cmd is None and not args:
                raise ValueError("cmd_vec or gimbal_rate_cmd is required.")
            cmd_vec = gimbal_rate_cmd if gimbal_rate_cmd is not None else args[0]
        if isinstance(cmd_vec, (int, float, np.floating)):
            g_rate_cmd = float(cmd_vec)
            w_torque_cmd = 0.0
        else:
            g_rate_cmd, w_torque_cmd = cmd_vec

        # Apply saturation
        g_rate = float(self.apply_saturation(g_rate_cmd))
        if self.max_wheel_torque is not None:
            w_torque = float(np.clip(w_torque_cmd, -self.max_wheel_torque, self.max_wheel_torque))
        else:
            w_torque = float(w_torque_cmd)

        # Update momentum magnitude for parent logic
        self.h_mag = self.inertia * self.wheel_speed

        # Current axes
        s, t = self.get_axes()

        # Net Torque = Torque_RW + Torque_CMG
        # T = w_torque * s + g_rate * (h * t)
        torque_vec = w_torque * s + g_rate * self.h_mag * t

        # Update states
        if dt is not None:
            self.gimbal_angle += g_rate * dt
            self.gimbal_angle = (self.gimbal_angle + np.pi) % (2 * np.pi) - np.pi
            self.wheel_speed += (w_torque / self.inertia) * dt

        return torque_vec

    def get_jacobian(self, angle: float | None = None) -> np.ndarray:
        """
        Returns full Jacobian matrix mapping [g_rate, w_torque]^T to torque vector.

        Parameters
        ----------
        angle : float, optional
            Gimbal angle to evaluate at (rad).

        Returns
        -------
        np.ndarray
            (3, 2) Jacobian matrix.
        """
        s, t = self.get_axes(angle)
        h = self.inertia * self.wheel_speed

        # T = [h*t , s] * [g_rate, w_torque]^T
        jac = np.zeros((3, 2))
        jac[:, 0] = h * t
        jac[:, 1] = s
        return jac
