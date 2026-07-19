"""Inertial Measurement Unit (IMU) with accelerometer and gyroscope channels."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.sensor import Sensor, SensorMeasurement


class Accelerometer(Sensor):
    """Accelerometer sensor model."""

    quantity = "specific_force"
    units = "m/s^2"
    frame = "body"

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: np.ndarray | None = None,
        scale_factor: float = 1.0,
        name: str = "Accelerometer",
    ) -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float) if bias is not None else np.zeros(3)
        self.scale_factor = scale_factor

    def measure(self, true_accel: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_accel is None:
            if not args:
                raise ValueError("true_accel is required.")
            true_accel = np.asarray(args[0])
        noise = np.random.normal(0, self.noise_std, 3)
        measured_accel = self.scale_factor * np.asarray(true_accel) + self.bias + noise
        return self.build_measurement(measured_accel)


class IMU(Sensor):
    """IMU combining gyroscope and accelerometer channels."""

    quantity = "imu"
    units = ("rad/s", "m/s^2")
    frame = "body"

    def __init__(
        self,
        gyro_params: dict | None = None,
        accel_params: dict | None = None,
        name: str = "IMU",
    ) -> None:
        super().__init__(name)
        self.gyro = Gyroscope(**(gyro_params or {}))
        self.accel = Accelerometer(**(accel_params or {}))

    def measure(
        self,
        true_omega: np.ndarray | None = None,
        true_accel: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SensorMeasurement:
        if true_omega is None:
            if not args:
                raise ValueError("true_omega is required.")
            true_omega = np.asarray(args[0])
        if true_accel is None:
            if len(args) < 2:
                raise ValueError("true_accel is required.")
            true_accel = np.asarray(args[1])
        gyro_packet = self.gyro.measure(true_omega, **kwargs)
        accel_packet = self.accel.measure(true_accel, **kwargs)
        covariance = np.block([
            [gyro_packet.covariance, np.zeros((3, 3))],
            [np.zeros((3, 3)), accel_packet.covariance],
        ])
        value = np.concatenate([gyro_packet.value, accel_packet.value])
        return self.build_measurement(value, covariance=covariance, metadata={"components": ["angular_rate", "specific_force"]})

    def measurement_noise_std(self) -> np.ndarray:
        return np.concatenate([self.gyro.measurement_noise_std(), self.accel.measurement_noise_std()])
