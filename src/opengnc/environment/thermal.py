"""
Spacecraft thermal environment flux models (Solar, Albedo, Earth IR).
"""

import numpy as np


class ThermalEnvironment:
    """
    Spacecraft External Thermal Environment Models.

    Calculates heat fluxes from direct solar radiation, planetary albedo, and
    outgoing longwave radiation (Earth IR).

    Parameters
    ----------
    albedo_coeff : float, optional
        Planetary mean spherical albedo [0, 1]. Default 0.3.
    earth_ir : float, optional
        Mean outgoing longwave radiation (W/m^2). Default 230.0.
    """

    def __init__(self, albedo_coeff: float = 0.3, earth_ir: float = 230.0) -> None:
        """Initialize thermal constants."""
        self.albedo_coeff = albedo_coeff
        self.earth_ir = earth_ir
        self.solar_constant = 1361.0  # W/m^2 at 1 AU

    def get_solar_flux(self, distance_au: float = 1.0) -> float:
        """
        Calculate direct solar flux.

        Equation: $F_{sun} = S / d^2$

        Parameters
        ----------
        distance_au : float, optional
            Sun-spacecraft distance (AU). Default 1.0.

        Returns
        -------
        float
            Solar flux ($W/m^2$).
        """
        return float(self.solar_constant / (float(distance_au)**2))

    def get_albedo_flux(self, r_sat: np.ndarray, r_sun: np.ndarray) -> float:
        """
        Calculate planet-reflected solar flux (Albedo).

        Uses a Lambertian reflection model.

        Parameters
        ----------
        r_sat : np.ndarray
            Satellite ECI position (m).
        r_sun : np.ndarray
            Sun ECI position (m).

        Returns
        -------
        float
            Albedo flux on a nadir-facing surface ($W/m^2$).
        """
        rs = np.asarray(r_sat)
        rn = np.asarray(r_sun)

        r_sat_u = rs / np.linalg.norm(rs)
        r_sun_u = rn / np.linalg.norm(rn)

        cos_zeta = np.dot(r_sat_u, r_sun_u)

        if cos_zeta < 0:
            return 0.0

        return float(self.solar_constant * self.albedo_coeff * cos_zeta)

    def get_earth_ir_flux(self) -> float:
        """
        Get standard Earth IR flux.

        Returns
        -------
        float
            Planetary IR flux ($W/m^2$).
        """
        return float(self.earth_ir)




