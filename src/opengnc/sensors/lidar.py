"""
Lidar sensor model.
"""

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor


class Lidar(Sensor):
    """
    Lidar sensor model.

    Measures range and line-of-sight (LOS) vector to a target point.

    Parameters
    ----------
    range_noise_std : float, optional
        Range measurement noise standard deviation (m). Default is 0.01.
    los_noise_std : float, optional
        Angular noise standard deviation for LOS vector (rad). Default is 0.001.
    name : str, optional
        Sensor name. Default is "Lidar".
    """

    def __init__(
        self,
        range_noise_std: float = 0.01,
        los_noise_std: float = 0.001,
        name: str = "Lidar",
    ) -> None:
        super().__init__(name)
        self.range_noise_std = range_noise_std
        self.los_noise_std = los_noise_std

    def measure(
        self, true_relative_pos: np.ndarray | None = None, *args: Any, **kwargs: Any
    ) -> tuple[float, np.ndarray]:
        """
        Generate Lidar range and LOS measurement.

        Parameters
        ----------
        true_relative_pos : np.ndarray
            True relative position vector in body frame (m).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        tuple[float, np.ndarray]
            (measured_range, measured_los_vec).
            measured_range : Range to target (m).
            measured_los_vec : Unit LOS vector in body frame.
        """
        if true_relative_pos is None:
            if not args:
                raise ValueError("true_relative_pos is required.")
            true_relative_pos = np.asarray(args[0])
        true_range = float(np.linalg.norm(true_relative_pos))
        true_los = true_relative_pos / true_range if true_range > 0 else np.zeros(3)

        # Add range noise
        measured_range = true_range + np.random.normal(0, self.range_noise_std)

        # Add LOS noise (small random rotation)
        if true_range > 0:
            noise_vec = np.random.normal(0, self.los_noise_std, 3)
            measured_los = true_los + noise_vec
            measured_los /= np.linalg.norm(measured_los)
        else:
            measured_los = true_los

        return float(max(0.0, measured_range)), measured_los




