"""Lidar sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class Lidar(Sensor):
    """Lidar sensor model."""

    quantity = "range_los"
    units = ("m", "unit_vector")
    frame = "body"

    def __init__(self, range_noise_std: float = 0.01, los_noise_std: float = 0.001, name: str = "Lidar") -> None:
        super().__init__(name)
        self.range_noise_std = range_noise_std
        self.los_noise_std = los_noise_std

    def measure(self, true_relative_pos: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_relative_pos is None:
            if not args:
                raise ValueError("true_relative_pos is required.")
            true_relative_pos = np.asarray(args[0])
        true_range = float(np.linalg.norm(true_relative_pos))
        true_los = true_relative_pos / true_range if true_range > 0 else np.zeros(3)
        measured_range = float(max(0.0, true_range + np.random.normal(0, self.range_noise_std)))
        if true_range > 0:
            noise_vec = np.random.normal(0, self.los_noise_std, 3)
            measured_los = true_los + noise_vec
            measured_los /= np.linalg.norm(measured_los)
        else:
            measured_los = true_los
        covariance = np.diag([
            self.range_noise_std**2,
            self.los_noise_std**2,
            self.los_noise_std**2,
            self.los_noise_std**2,
        ])
        return self.build_measurement(np.concatenate([[measured_range], measured_los]), covariance=covariance, metadata={"components": ["range", "line_of_sight"]})

    def measurement_noise_std(self) -> np.ndarray:
        return np.array([self.range_noise_std, self.los_noise_std, self.los_noise_std, self.los_noise_std], dtype=float)
