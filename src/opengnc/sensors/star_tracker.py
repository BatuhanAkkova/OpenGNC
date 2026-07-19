"""Star tracker sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement
from opengnc.utils.quat_utils import quat_mult, quat_normalize


class StarTracker(Sensor):
    """Star tracker attitude sensor."""

    quantity = "attitude_quaternion"
    units = "quaternion"
    frame = "body_to_inertial"

    def __init__(self, noise_std: float = 0.0, bias: np.ndarray | None = None, name: str = "StarTracker") -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float) if bias is not None else np.zeros(3)

    def measure(self, true_quat: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_quat is None:
            if not args:
                raise ValueError("true_quat is required.")
            true_quat = np.asarray(args[0])
        true_quaternion = quat_normalize(np.asarray(true_quat, dtype=float))
        noise = np.random.normal(0, self.noise_std, 3)
        error_vec = self.bias + noise
        angle = np.linalg.norm(error_vec)
        if angle > 1e-8:
            axis = error_vec / angle
            q_err = np.array([
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
                np.cos(angle / 2),
            ])
        else:
            q_err = np.array([0.0, 0.0, 0.0, 1.0])
        q_meas = quat_normalize(np.asarray(self.apply_faults(quat_mult(true_quaternion, q_err)), dtype=float))
        covariance = np.eye(3) * float(self.noise_std**2)
        return self.build_measurement(q_meas, covariance=covariance, metadata={"error_parameterization": "small_angle_vector"})

    def measurement_noise_std(self) -> np.ndarray:
        return np.full(3, float(self.noise_std), dtype=float)
