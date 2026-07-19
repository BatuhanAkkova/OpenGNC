"""Sensor factory helpers for configuration-driven sensor suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opengnc.sensors.altimeter import Altimeter
from opengnc.sensors.camera import Camera
from opengnc.sensors.gnss_receiver import GNSSReceiver
from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.horizon_sensor import HorizonSensor
from opengnc.sensors.imu import IMU, Accelerometer
from opengnc.sensors.lidar import Lidar
from opengnc.sensors.magnetometer import Magnetometer
from opengnc.sensors.sensor import Sensor
from opengnc.sensors.star_tracker import StarTracker
from opengnc.sensors.sun_sensor import SunSensor
from opengnc.sensors.sun_sensor_array import CoarseSunSensorArray
from opengnc.simulation.scenario import ScenarioConfig

SENSOR_REGISTRY: dict[str, type[Sensor]] = {
    "accelerometer": Accelerometer,
    "altimeter": Altimeter,
    "camera": Camera,
    "coarse_sun_sensor_array": CoarseSunSensorArray,
    "gnss": GNSSReceiver,
    "gnssreceiver": GNSSReceiver,
    "gyroscope": Gyroscope,
    "horizon_sensor": HorizonSensor,
    "imu": IMU,
    "lidar": Lidar,
    "magnetometer": Magnetometer,
    "star_tracker": StarTracker,
    "sun_sensor": SunSensor,
}


def _convert_value(value: Any) -> Any:
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return np.asarray(value, dtype=float)
        return [_convert_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _convert_value(val) for key, val in value.items()}
    return value


def build_sensor(sensor_type: str, params: dict[str, Any] | None = None) -> Sensor:
    """Instantiate a sensor from its registry name and parameter dictionary."""
    key = sensor_type.strip().lower()
    if key not in SENSOR_REGISTRY:
        raise KeyError(f"Unsupported sensor type: {sensor_type}")
    normalized = _convert_value(params or {})
    return SENSOR_REGISTRY[key](**normalized)


def load_sensor_suite(config: str | Path | dict[str, Any]) -> list[Sensor]:
    """Load a list of sensors from a config dictionary or JSON/YAML file."""
    if isinstance(config, (str, Path)):
        data = ScenarioConfig(config).config
    else:
        data = config

    sensors_cfg = data.get("sensors", [])
    sensors: list[Sensor] = []
    for entry in sensors_cfg:
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError("Each sensor entry must contain at least a 'type' field.")
        sensor_type = str(entry["type"])
        params = {key: value for key, value in entry.items() if key != "type"}
        sensors.append(build_sensor(sensor_type, params))
    return sensors
