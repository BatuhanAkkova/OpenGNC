"""Sensor-agnostic attitude fusion wrapper built on MEKF or UKF_Attitude."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.kalman_filters.mekf import MEKF
from opengnc.kalman_filters.ukf import UKF_Attitude
from opengnc.sensors.sensor import SensorMeasurement


class AttitudeSensorFusion:
    """Decouple attitude filters from specific sensor combinations."""

    def __init__(self, backend: str = "mekf", **filter_kwargs: Any) -> None:
        backend_key = backend.strip().lower()
        self.filter: MEKF | UKF_Attitude
        if backend_key == "mekf":
            self.filter = MEKF(**filter_kwargs)
        elif backend_key in {"ukf", "ukf_attitude"}:
            self.filter = UKF_Attitude(**filter_kwargs)
        else:
            raise ValueError(f"Unsupported attitude fusion backend: {backend}")
        self.backend = backend_key

    @property
    def quaternion(self) -> np.ndarray:
        if hasattr(self.filter, "q"):
            return np.asarray(self.filter.q, dtype=float)
        return np.asarray(self.filter.x[:4], dtype=float)

    @property
    def bias(self) -> np.ndarray:
        if hasattr(self.filter, "beta"):
            return np.asarray(self.filter.beta, dtype=float)
        return np.asarray(self.filter.x[4:7], dtype=float)

    def predict(self, measurement: SensorMeasurement, dt: float | None = None) -> None:
        """Propagate the attitude state from an angular-rate sensor packet."""
        self.filter.predict(measurement, dt=dt)

    def update_from_measurement(self, measurement: SensorMeasurement) -> None:
        if measurement.quantity == "angular_rate":
            raise ValueError("Angular-rate measurements belong in the predict step, not the update step.")
        self.filter.update(measurement)

    def update_from_measurements(self, measurements: list[SensorMeasurement]) -> None:
        for measurement in measurements:
            self.update_from_measurement(measurement)
