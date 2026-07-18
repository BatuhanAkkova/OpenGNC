"""
Abstract base class for actuator models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Actuator(ABC):
    """
    Abstract base class for all actuators.

    Parameters
    ----------
    name : str, optional
        Actuator name. Default is "Actuator".
    saturation : float | tuple[float, float], optional
        Max output magnitude (scalar) or (min, max) range.
    deadband : float, optional
        Input values below this magnitude result in zero output.
    """

    def __init__(
        self,
        name: str = "Actuator",
        saturation: float | tuple[float, float] | None = None,
        deadband: float | None = None,
    ) -> None:
        r"""
        Initialize actuator base.

        Parameters
        ----------
        name : str, optional
            Actuator name. Default "Actuator".
        saturation : float | tuple[float, float] | None, optional
            Saturation limit. Scalar for symmetric $[\text{-limit}, \text{limit}]$
            or tuple for $[\min, \max]$.
        deadband : float | None, optional
            Deadband threshold.
        """
        self.name = name
        self.saturation = saturation
        self.deadband = deadband

    @abstractmethod
    def command(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calculate the actuator output based on the command signal.

        Parameters
        ----------
        *args : Any
            Commanded input(s) (e.g., torque, dipole, voltage).
        **kwargs : dict
            Additional state info (e.g., current speed, environment).

        Returns
        -------
        Any
            The actual output applied to the system.
        """
        pass

    def apply_saturation(self, value: np.ndarray | float) -> np.ndarray | float:
        """
        Apply saturation limits.

        Parameters
        ----------
        value : np.ndarray | float
            Signal to saturate.

        Returns
        -------
        np.ndarray | float
            Saturated signal.
        """
        if self.saturation is None:
            return value

        if isinstance(self.saturation, (int, float)):
            limit = abs(float(self.saturation))
            return np.clip(value, -limit, limit)
        elif isinstance(self.saturation, (list, tuple)) and len(self.saturation) == 2:
            return np.clip(value, self.saturation[0], self.saturation[1])
        return value

    def apply_deadband(self, value: float) -> float:
        """
        Apply deadband to value.

        Parameters
        ----------
        value : float
            Commanded value.

        Returns
        -------
        float
            Value after deadband application.
        """
        if self.deadband is None or self.deadband == 0:
            return value

        if abs(value) < self.deadband:
            return 0.0
        return float(value)




