"""Earth / horizon sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class HorizonSensor(Sensor):
    """Earth / horizon sensor model."""

    quantity = "nadir_vector"
    units = "unit_vector"
    frame = "body"

    def __init__(self, noise_std: float = 0.01, bias: np.ndarray | None = None, name: str = "HorizonSensor") -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float) if bias is not None else np.zeros(2)

    def measure(self, true_nadir_vec: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_nadir_vec is None:
            if not args:
                raise ValueError("true_nadir_vec is required.")
            true_nadir_vec = np.asarray(args[0])
        nadir = true_nadir_vec / np.linalg.norm(true_nadir_vec)
        meas_n = nadir + np.random.normal(0, self.noise_std, 3)
        if np.linalg.norm(self.bias) > 0:
            meas_n[0] += self.bias[0]
            meas_n[1] += self.bias[1]
        meas_n = np.asarray(meas_n / np.linalg.norm(meas_n), dtype=float)
        return self.build_measurement(meas_n)

    def measurement_bias(self) -> np.ndarray | None:
        return np.array([self.bias[0], self.bias[1], 0.0], dtype=float)
