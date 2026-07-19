"""Gyroscope sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class Gyroscope(Sensor):
    """Gyroscope sensor model."""

    quantity = "angular_rate"
    units = "rad/s"
    frame = "body"

    def __init__(
        self,
        noise_std: float = 0.0,
        bias_stability: float = 0.0,
        initial_bias: np.ndarray | None = None,
        dt: float = 0.1,
        misalignment: np.ndarray | None = None,
        scale_factor: float | np.ndarray = 1.0,
        name: str = "Gyroscope",
    ) -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias_stability = bias_stability
        self.current_bias = np.asarray(initial_bias, dtype=float) if initial_bias is not None else np.zeros(3)
        self.dt = dt
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(self, true_omega: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_omega is None:
            if not args:
                raise ValueError("true_omega is required.")
            true_omega = np.asarray(args[0])
        dt = kwargs.get("dt", self.dt)
        if self.bias_stability > 0:
            walk_std = self.bias_stability * np.sqrt(dt)
            self.current_bias += np.random.normal(0, walk_std, 3)
        omega_cal = self.apply_calibration(true_omega, self.misalignment, self.scale_factor)
        measurement_noise = np.random.normal(0, self.noise_std, 3)
        measured_omega = np.asarray(self.apply_faults(omega_cal + self.current_bias + measurement_noise), dtype=float)
        return self.build_measurement(measured_omega, metadata={"sample_period_s": dt, "bias_random_walk_std": self.bias_stability})
