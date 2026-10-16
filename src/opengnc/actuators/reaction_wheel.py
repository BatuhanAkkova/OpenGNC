"""
Reaction Wheel (RW) actuator model with friction and momentum limits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.actuators.actuator import Actuator


class ReactionWheel(Actuator):
    """
    Reaction Wheel Actuator.

    Models torque generation with saturation, max speed limits, and friction.

    Parameters
    ----------
    max_torque : float, optional
        Maximum torque (Nm). Default is None (no saturation).
    max_momentum : float, optional
        Maximum angular momentum (Nms).
    inertia : float, optional
        Moment of inertia about spin axis (kg*m^2).
    name : str, optional
        Actuator name. Default is "RW".
    static_friction : float, optional
        Static friction torque (Nm). Default is 0.0.
    viscous_friction : float, optional
        Viscous friction coefficient (Nm/(rad/s)). Default is 0.0.
    coulomb_friction : float, optional
        Coulomb friction torque (Nm). Default is 0.0.
    """

    def __init__(
        self,
        max_torque: float | None = None,
        max_momentum: float | None = None,
        inertia: float | None = None,
        name: str = "RW",
        static_friction: float = 0.0,
        viscous_friction: float = 0.0,
        coulomb_friction: float = 0.0,
    ) -> None:
        r"""
        Initialize reaction wheel parameters.

        Parameters
        ----------
        max_torque : float | None, optional
            Saturation torque (Nm).
        max_momentum : float | None, optional
            Momentum storage limit (Nms).
        inertia : float | None, optional
            Wheel inertia ($kg \cdot m^2$).
        name : str, optional
            Actuator name. Default "RW".
        static_friction : float, optional
            Static friction torque (Nm).
        viscous_friction : float, optional
            Viscous coefficient ($Nm / (rad/s)$).
        coulomb_friction : float, optional
            Coulomb friction torque (Nm).
        """
        super().__init__(name=name, saturation=max_torque)
        self.max_momentum = max_momentum
        self.inertia = inertia
        self.static_friction = static_friction
        self.viscous_friction = viscous_friction
        self.coulomb_friction = coulomb_friction

    def command(
        self, torque_cmd: float | None = None, current_speed: float = 0.0, *args: Any, **kwargs: Any
    ) -> float:
        r"""
        Calculate delivered motor torque.

        Friction Model:
        $\tau_f = f_v \omega + f_c \text{sgn}(\omega)$

        Parameters
        ----------
        torque_cmd : float
            Commanded motor torque (Nm).
        current_speed : float, optional
            Current wheel spin rate (rad/s). Default 0.0.

        Returns
        -------
        float
            Actual torque delivered to the spacecraft (Nm).
        """
        if torque_cmd is None:
            if not args:
                raise ValueError("torque_cmd is required.")
            torque_cmd = float(args[0])
        torque = self.apply_deadband(torque_cmd)
        torque = float(self.apply_saturation(torque))

        if abs(current_speed) < 1e-4:
            if abs(torque) < self.static_friction:
                torque = 0.0
            else:
                torque -= float(np.sign(torque) * self.static_friction)
        else:
            friction_torque = (
                self.viscous_friction * current_speed
                + self.coulomb_friction * np.sign(current_speed)
            )
            torque -= float(friction_torque)

        if self.inertia is not None and self.max_momentum is not None:
            current_momentum = float(self.inertia * current_speed)

            if (current_momentum >= self.max_momentum and torque > 0) or (
                current_momentum <= -self.max_momentum and torque < 0
            ):
                torque = 0.0

        return float(torque)
