"""Sun sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class SunSensor(Sensor):
    """Sun sensor model."""

    quantity = "sun_vector"
    units = "unit_vector"
    frame = "body"

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: np.ndarray | None = None,
        misalignment: np.ndarray | None = None,
        scale_factor: float | np.ndarray = 1.0,
        name: str = "SunSensor",
    ) -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float) if bias is not None else np.zeros(3)
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(self, true_sun_vec_body: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_sun_vec_body is None:
            if not args:
                raise ValueError("true_sun_vec_body is required.")
            true_sun_vec_body = np.asarray(args[0])
        calibrated = self.apply_calibration(true_sun_vec_body, self.misalignment, self.scale_factor, self.bias)
        measured_vec = np.asarray(self.apply_faults(self.add_gaussian_noise(calibrated, self.noise_std)), dtype=float)
        norm = np.linalg.norm(measured_vec)
        if norm > 0:
            measured_vec = measured_vec / norm
        return self.build_measurement(measured_vec)
