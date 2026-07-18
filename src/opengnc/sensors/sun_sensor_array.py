"""
Coarse Sun Sensor Array model.
"""

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor


class CoarseSunSensorArray(Sensor):
    r"""
    Array of Coarse Sun Sensors (CSS).

    Each CSS measures the cosine of the angle between the sun vector and its boresight.
    $I = I_{max} \cos(\theta) = I_{max} (S \cdot N)$

    Parameters
    ----------
    boresights : list[np.ndarray], optional
        Unit vectors representing the boresight direction of each CSS in the body frame.
        Default is 6 faces of a cube.
    i_max : float, optional
        Maximum current/output when sun is aligned with boresight. Default is 1.0.
    noise_std : float, optional
        Standard deviation of measurement noise. Default is 0.01.
    name : str, optional
        Sensor name. Default is "CSSArray".
    """

    def __init__(
        self,
        boresights: list[np.ndarray] | None = None,
        i_max: float = 1.0,
        noise_std: float = 0.01,
        name: str = "CSSArray",
    ) -> None:
        super().__init__(name)
        if boresights is None:
            # Default to 6 faces of a cube
            boresights_list = [
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        else:
            boresights_list = boresights

        self.boresights = [b / np.linalg.norm(b) for b in boresights_list]
        self.i_max = i_max
        self.noise_std = noise_std

    def measure(
        self, true_sun_vec: np.ndarray | None = None, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Generate measurements from each CSS in the array.

        Parameters
        ----------
        true_sun_vec : np.ndarray
            True sun unit vector in body frame.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray
            Array of measurements (typically current or voltage) from each CSS.
        """
        if true_sun_vec is None:
            if not args:
                raise ValueError("true_sun_vec is required.")
            true_sun_vec = np.asarray(args[0])
        s = true_sun_vec / np.linalg.norm(true_sun_vec)
        measurements = []
        for n in self.boresights:
            cos_theta = float(np.dot(s, n))
            # CSS only works if sun is in front of the sensor
            i_meas = self.i_max * max(0.0, cos_theta)
            i_meas += np.random.normal(0, self.noise_std)
            measurements.append(float(max(0.0, i_meas)))

        return np.array(measurements)




