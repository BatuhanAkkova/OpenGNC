"""
Simple pinhole camera model.
"""

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor


class Camera(Sensor):
    """
    Simple pinhole camera model.

    Projects 3D points in the body frame onto a 2D image plane.

    Parameters
    ----------
    focal_length : float, optional
        Focal length (m). Default is 1.0.
    resolution : tuple[int, int], optional
        (width, height) in pixels. Default is (1024, 1024).
    sensor_size : tuple[float, float], optional
        (width, height) in physical units (e.g., m). Default is (1.0, 1.0).
    noise_std : float, optional
        Pixel noise standard deviation. Default is 0.0.
    name : str, optional
        Sensor name. Default is "Camera".
    """

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

        # Pixels per unit distance
        self.sx = resolution[0] / sensor_size[0]
        self.sy = resolution[1] / sensor_size[1]

        # Principal point (center of image)
        self.cx = resolution[0] / 2
        self.cy = resolution[1] / 2

    def measure(
        self, true_point_body: np.ndarray | None = None, *args: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """
        Project 3D point onto image plane.

        Parameters
        ----------
        true_point_body : np.ndarray
            3D point in the camera/body frame (m).
            Assumes Z is along the optical axis.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray | None
            (u, v) pixel coordinates, or None if outside FOV or behind camera.
        """
        if true_point_body is None:
            if not args:
                raise ValueError("true_point_body is required.")
            true_point_body = np.asarray(args[0])
        x, y, z = true_point_body

        if z <= 0:
            return None  # Point is behind the camera

        # Pinhole projection: u = f*x/z, v = f*y/z
        u = (self.focal_length * x / z) * self.sx + self.cx
        v = (self.focal_length * y / z) * self.sy + self.cy

        # Add pixel noise
        if self.noise_std > 0:
            u += np.random.normal(0, self.noise_std)
            v += np.random.normal(0, self.noise_std)

        # Check if point is within image boundaries
        if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
            return np.array([u, v])

        return None




