"""
Abstract sensor interfaces and standardized measurement containers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class SensorNoiseModel:
    """Describe a sensor's stochastic error convention."""

    model: str
    std_dev: np.ndarray
    bias: np.ndarray | None = None
    correlation_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def covariance(self) -> np.ndarray:
        """Return a diagonal covariance matrix derived from ``std_dev``."""
        sigma = np.atleast_1d(np.asarray(self.std_dev, dtype=float))
        return np.diag(np.square(sigma))


@dataclass(frozen=True)
class SensorMeasurement:
    """Standardized measurement packet emitted by all sensors."""

    sensor_name: str
    quantity: str
    value: np.ndarray
    units: str | tuple[str, ...]
    frame: str | None
    covariance: np.ndarray
    noise_model: SensorNoiseModel
    metadata: dict[str, Any] = field(default_factory=dict)


class Sensor(ABC):
    """Abstract base class for all sensors."""

    quantity = "measurement"
    units: str | tuple[str, ...] = ""
    frame: str | None = None
    noise_model_name = "gaussian"

    def __init__(self, name: str = "Sensor") -> None:
        self.name = name
        self.fault_state: str | None = None
        self.stuck_value: np.ndarray | float | None = None

    @abstractmethod
    def measure(self, *args: Any, **kwargs: Any) -> SensorMeasurement:
        """Generate a standardized measurement packet."""
        raise NotImplementedError

    def build_measurement(
        self,
        raw_value: Any,
        *,
        covariance: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        frame: str | None = None,
    ) -> SensorMeasurement:
        """Build a standardized measurement packet from a raw value."""
        return SensorMeasurement(
            sensor_name=self.name,
            quantity=self.quantity,
            value=self._flatten_value(raw_value),
            units=self.units,
            frame=self.frame if frame is None else frame,
            covariance=self.measurement_covariance(raw_value) if covariance is None else covariance,
            noise_model=self.noise_model(),
            metadata=self.measurement_metadata(raw_value) | (metadata or {}),
        )

    def measurement_metadata(self, value: Any) -> dict[str, Any]:
        return {}

    def measurement_noise_std(self) -> np.ndarray:
        sigma = getattr(self, "noise_std", 0.0)
        return self._flatten_value(sigma)

    def measurement_bias(self) -> np.ndarray | None:
        for attr in ("bias", "current_bias", "pos_bias", "vel_bias"):
            if hasattr(self, attr):
                return self._flatten_value(getattr(self, attr))
        return None

    def noise_model(self) -> SensorNoiseModel:
        metadata: dict[str, Any] = {}
        if hasattr(self, "bias_stability"):
            metadata["bias_stability"] = float(getattr(self, "bias_stability"))
        if hasattr(self, "dt"):
            metadata["sample_period_s"] = float(getattr(self, "dt"))
        return SensorNoiseModel(
            model=self.noise_model_name,
            std_dev=self.measurement_noise_std(),
            bias=self.measurement_bias(),
            correlation_time=getattr(self, "correlation_time", None),
            metadata=metadata,
        )

    def measurement_covariance(self, value: Any) -> np.ndarray:
        sigma = self.measurement_noise_std()
        if sigma.size == 1:
            dim = self._flatten_value(value).size
            return np.eye(dim) * float(sigma[0] ** 2)
        if sigma.size != self._flatten_value(value).size:
            raise ValueError(
                f"Noise specification for {self.name} has size {sigma.size}, "
                f"but the measurement has size {self._flatten_value(value).size}."
            )
        return np.diag(np.square(sigma))

    def apply_calibration(
        self,
        value: np.ndarray | float,
        misalignment: np.ndarray | None = None,
        scale_factor: np.ndarray | float = 1.0,
        bias: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        if isinstance(value, np.ndarray):
            if misalignment is not None:
                value = (np.eye(len(value)) + misalignment) @ value
            value = scale_factor * value
            if bias is not None:
                value = value + bias
        else:
            value = scale_factor * value
            if bias is not None:
                value += bias
        return cast(np.ndarray | float, value)

    def apply_fogm_noise(
        self, current_val: np.ndarray | float, sigma: float, tau: float, dt: float
    ) -> np.ndarray | float:
        if sigma == 0 or tau <= 0:
            return cast(np.ndarray | float, current_val)
        phi = np.exp(-dt / tau)
        q = sigma * np.sqrt(1 - np.exp(-2 * dt / tau))
        noise = np.random.normal(0, q, size=np.shape(current_val))
        return cast(np.ndarray | float, phi * current_val + noise)

    def apply_faults(self, value: np.ndarray | float) -> np.ndarray | float:
        if self.fault_state == "stuck":
            return cast(np.ndarray | float, self.stuck_value if self.stuck_value is not None else value)
        if self.fault_state == "spike":
            spike = np.random.normal(0, 100 * np.std(value) if np.std(value) > 0 else 10.0, size=np.shape(value))
            return cast(np.ndarray | float, value + spike)
        if self.fault_state == "noise_increase":
            return cast(
                np.ndarray | float,
                value + np.random.normal(0, 10.0 * np.std(value) if np.std(value) > 0 else 1.0, size=np.shape(value)),
            )
        return cast(np.ndarray | float, value)

    def add_gaussian_noise(self, value: np.ndarray | float, std_dev: float) -> np.ndarray | float:
        if std_dev is None or std_dev == 0:
            return cast(np.ndarray | float, value)
        noise = np.random.normal(0, std_dev, size=np.shape(value))
        return cast(np.ndarray | float, value + noise)

    @staticmethod
    def _flatten_value(value: Any) -> np.ndarray:
        if isinstance(value, (tuple, list)):
            parts = [Sensor._flatten_value(part) for part in value]
            return np.concatenate(parts) if parts else np.array([], dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr.reshape(-1)
