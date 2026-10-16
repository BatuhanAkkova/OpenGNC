"""
Magnetorquer actuator model.
"""

from __future__ import annotations

from typing import Any

from opengnc.actuators.actuator import Actuator


class Magnetorquer(Actuator):
    """
    Magnetorquer Actuator Model.

    Parameters
    ----------
    max_dipole : float | None, optional
        Maximum dipole moment (Am^2). Default None (no saturation).
    name : str, optional
        Actuator name. Default "MTQ".
    """

    def __init__(self, max_dipole: float | None = None, name: str = "MTQ") -> None:
        """Initialize magnetorquer."""
        super().__init__(name=name, saturation=max_dipole)

    def command(self, dipole_cmd: float | None = None, *args: Any, **kwargs: Any) -> float:
        """
        Calculate delivered dipole moment.

        Parameters
        ----------
        dipole_cmd : float
            Commanded dipole moment (Am^2).
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        float
            Delivered dipole moment (Am^2).
        """
        # Apply saturation
        if dipole_cmd is None:
            if not args:
                raise ValueError("dipole_cmd is required.")
            dipole_cmd = float(args[0])
        return float(self.apply_saturation(dipole_cmd))
