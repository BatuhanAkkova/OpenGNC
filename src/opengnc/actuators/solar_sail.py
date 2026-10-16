"""
Solar Sail force model based on Solar Radiation Pressure (SRP).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from opengnc.actuators.actuator import Actuator


class SolarSail(Actuator):
    """
    Solar Sail Actuator model.

    Models the force produced by Solar Radiation Pressure (SRP) on a flat sail surface.

    Parameters
    ----------
    area : float
        Surface area of the sail (m^2).
    reflectivity : float, optional
        Total reflectivity coefficient [0 to 1]. Default is 0.9.
    specular_reflect_coeff : float, optional
        Fraction of reflection that is specular [0 to 1]. Default is 0.9.
    name : str, optional
        Actuator name. Default is "SolarSail".
    """

    def __init__(
        self,
        area: float,
        reflectivity: float = 0.9,
        specular_reflect_coeff: float = 0.9,
        name: str = "SolarSail",
    ) -> None:
        super().__init__(name=name)
        self.area = area
        self.rho = reflectivity
        self.s_coeff = specular_reflect_coeff
        self.P0 = 4.56e-6  # Solar radiation pressure at 1 AU [N/m^2]

    def calculate_force(
        self, sun_unit_vec: np.ndarray, normal_unit_vec: np.ndarray, distance_au: float = 1.0
    ) -> np.ndarray:
        """
        Calculate SRP force vector on the flat sail surface.

        Parameters
        ----------
        sun_unit_vec : np.ndarray
            Unit vector from sail towards the Sun.
        normal_unit_vec : np.ndarray
            Unit vector normal to the sail surface.
        distance_au : float, optional
            Distance from the Sun in Astronomical Units (AU). Default is 1.0.

        Returns
        -------
        np.ndarray
            SRP force vector (N) in the same frame as input vectors.
        """
        u_sun = sun_unit_vec / np.linalg.norm(sun_unit_vec)
        n = normal_unit_vec / np.linalg.norm(normal_unit_vec)

        cos_theta = float(np.dot(u_sun, n))
        if cos_theta < 0:
            # Sun is behind the sail, no force (assuming opaque back for now)
            return np.zeros(3)

        # P = P0 / d^2
        P = self.P0 / (distance_au**2)

        # Flat-plate SRP: F = P*A*cos(theta)*[(1-rho_s)*u_sun + 2*(rho_s*cos(theta) + rho_d/3)*n]
        rho_s = self.rho * self.s_coeff
        rho_d = self.rho * (1 - self.s_coeff)

        force = (
            P
            * self.area
            * cos_theta
            * ((1 - rho_s) * u_sun + (2 * rho_s * cos_theta + (2 / 3) * rho_d) * n)
        )

        return cast(np.ndarray, force)

    def command(
        self, normal_cmd: np.ndarray | None = None, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate force based on commanded normal vector and current environment.

        Parameters
        ----------
        normal_cmd : np.ndarray
            Commanded normal vector for the sail.
        **kwargs : dict
            - sun_vec : np.ndarray
                Unit vector towards the Sun. Required.
            - distance_au : float, optional
                Distance from Sun (AU). Default is 1.0.

        Returns
        -------
        np.ndarray
            Force vector (N).
        """
        if normal_cmd is None:
            if not args:
                raise ValueError("normal_cmd is required.")
            normal_cmd = np.asarray(args[0])
        sun_vec = kwargs.get("sun_vec", np.array([1.0, 0.0, 0.0]))
        dist = float(kwargs.get("distance_au", 1.0))
        return cast(np.ndarray, self.calculate_force(sun_vec, normal_cmd, distance_au=dist))
