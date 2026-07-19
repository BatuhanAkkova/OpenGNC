"""Coarse sun sensor array model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class CoarseSunSensorArray(Sensor):
    """Array of coarse sun sensors (CSS)."""

    quantity = "sun_intensity_array"
    units = "sensor_unit"
    frame = "body"

    def __init__(
        self,
        boresights: list[np.ndarray] | None = None,
        i_max: float = 1.0,
        noise_std: float = 0.01,
        name: str = "CSSArray",
    ) -> None:
        super().__init__(name)
        if boresights is None:
            boresights_list = [
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        else:
            boresights_list = boresights
        self.boresights = [b / np.linalg.norm(b) for b in boresights_list]
        self.i_max = i_max
        self.noise_std = noise_std

    def measure(self, true_sun_vec: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_sun_vec is None:
            if not args:
                raise ValueError("true_sun_vec is required.")
            true_sun_vec = np.asarray(args[0])
        sun_unit = true_sun_vec / np.linalg.norm(true_sun_vec)
        measurements = []
        for boresight in self.boresights:
            cos_theta = float(np.dot(sun_unit, boresight))
            i_meas = self.i_max * max(0.0, cos_theta)
            i_meas += np.random.normal(0, self.noise_std)
            measurements.append(float(max(0.0, i_meas)))
        values = np.array(measurements)
        return self.build_measurement(values, metadata={"boresights": [b.copy() for b in self.boresights], "i_max": self.i_max})

    def measurement_noise_std(self) -> np.ndarray:
        return np.full(len(self.boresights), float(self.noise_std), dtype=float)
