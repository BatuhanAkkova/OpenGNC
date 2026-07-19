"""Simple pinhole camera model."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor, SensorMeasurement


class Camera(Sensor):
    """Simple pinhole camera model."""

    quantity = "pixel_coordinates"
    units = "px"
    frame = "image_plane"

    def __init__(
        self,
        focal_length: float = 1.0,
        resolution: tuple[int, int] = (1024, 1024),
        sensor_size: tuple[float, float] = (1.0, 1.0),
        noise_std: float = 0.0,
        name: str = "Camera",
    ) -> None:
        super().__init__(name)
        self.focal_length = focal_length
        self.resolution = resolution
        self.sensor_size = sensor_size
        self.noise_std = noise_std
        self.sx = resolution[0] / sensor_size[0]
        self.sy = resolution[1] / sensor_size[1]
        self.cx = resolution[0] / 2
        self.cy = resolution[1] / 2

    def measure(self, true_point_body: np.ndarray | None = None, *args: Any, **kwargs: Any) -> SensorMeasurement:
        if true_point_body is None:
            if not args:
                raise ValueError("true_point_body is required.")
            true_point_body = np.asarray(args[0])
        x_coord, y_coord, z_coord = true_point_body
        if z_coord <= 0:
            raise ValueError("Point is behind the camera.")
        u_coord = (self.focal_length * x_coord / z_coord) * self.sx + self.cx
        v_coord = (self.focal_length * y_coord / z_coord) * self.sy + self.cy
        if self.noise_std > 0:
            u_coord += np.random.normal(0, self.noise_std)
            v_coord += np.random.normal(0, self.noise_std)
        if not (0 <= u_coord < self.resolution[0] and 0 <= v_coord < self.resolution[1]):
            raise ValueError("Point projects outside the image plane.")
        return self.build_measurement(np.array([u_coord, v_coord]))
