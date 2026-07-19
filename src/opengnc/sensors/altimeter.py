"""Radar / altimeter sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class Altimeter(Sensor):
    """Radar / altimeter sensor model."""

    quantity = "altitude"
    units = "m"
    frame = "local_vertical"

    def __init__(self, noise_std: float = 1.0, bias: float = 0.0, name: str = "Altimeter") -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias

    def measure(self, true_altitude: float | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_altitude is None:
            if not args:
                raise ValueError("true_altitude is required.")
            true_altitude = float(args[0])
        measured_alt = float(max(0.0, true_altitude + self.bias + np.random.normal(0, self.noise_std)))
        return self.build_measurement(measured_alt)
