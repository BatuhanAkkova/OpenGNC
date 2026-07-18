"""
Radar / Altimeter sensor model.
"""

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor


class Altimeter(Sensor):
    """
    Radar / Altimeter sensor model.

    Measures height above a reference surface (altitude).

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of measurement noise (m). Default is 1.0.
    bias : float, optional
        Constant bias in altitude (m). Default is 0.0.
    name : str, optional
        Sensor name. Default is "Altimeter".
    """

    def __init__(self, noise_std: float = 1.0, bias: float = 0.0, name: str = "Altimeter") -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias

    def measure(self, true_altitude: float | None = None, *args: Any, **kwargs: Any) -> float:
        """
        Generate altitude measurement.

        Parameters
        ----------
        true_altitude : float
            True altitude above surface (m).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            Measured altitude (m). Guaranteed to be non-negative.
        """
        if true_altitude is None:
            if not args:
                raise ValueError("true_altitude is required.")
            true_altitude = float(args[0])
        measured_alt = true_altitude + self.bias + np.random.normal(0, self.noise_std)
        return float(max(0.0, measured_alt))




