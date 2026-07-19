"""GNSS receiver sensor model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class GNSSReceiver(Sensor):
    """GNSS receiver sensor model."""

    quantity = "position_velocity"
    units = ("m", "m/s")
    frame = "state_frame"

    def __init__(
        self,
        pos_noise_std: float = 10.0,
        vel_noise_std: float = 0.1,
        name: str = "GNSS",
        pos_bias: np.ndarray | None = None,
        vel_bias: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name)
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.pos_bias = np.asarray(pos_bias, dtype=float) if pos_bias is not None else np.zeros(3)
        self.vel_bias = np.asarray(vel_bias, dtype=float) if vel_bias is not None else np.zeros(3)

    def measure(
        self,
        true_pos: np.ndarray | None = None,
        true_vel: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SensorMeasurement:
        if true_pos is None:
            if not args:
                raise ValueError("true_pos is required.")
            true_pos = np.asarray(args[0])
        if true_vel is None:
            if len(args) < 2:
                raise ValueError("true_vel is required.")
            true_vel = np.asarray(args[1])
        meas_pos = np.asarray(self.apply_faults(self.add_gaussian_noise(true_pos, self.pos_noise_std) + self.pos_bias), dtype=float)
        meas_vel = np.asarray(self.apply_faults(self.add_gaussian_noise(true_vel, self.vel_noise_std) + self.vel_bias), dtype=float)
        covariance = np.diag([
            self.pos_noise_std**2,
            self.pos_noise_std**2,
            self.pos_noise_std**2,
            self.vel_noise_std**2,
            self.vel_noise_std**2,
            self.vel_noise_std**2,
        ])
        frame = kwargs.get("frame", self.frame)
        return self.build_measurement(np.concatenate([meas_pos, meas_vel]), covariance=covariance, frame=frame, metadata={"components": ["position", "velocity"], "frame": frame})

    def measurement_noise_std(self) -> np.ndarray:
        return np.array([self.pos_noise_std, self.pos_noise_std, self.pos_noise_std, self.vel_noise_std, self.vel_noise_std, self.vel_noise_std], dtype=float)

    def measurement_bias(self) -> np.ndarray | None:
        return np.concatenate([self.pos_bias, self.vel_bias])
