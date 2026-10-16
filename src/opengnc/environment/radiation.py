"""
Radiation environment models for TID and SEU rates estimation.
"""

import numpy as np


class RadiationModel:
    """
    Parametric Space Radiation Environment Models.

    Estimates Total Ionizing Dose (TID) and Single Event Upset (SEU) rates
    for spacecraft electronic components in Low Earth Orbit (LEO).
    """

    def __init__(self) -> None:
        """Initialize radiation model."""
        pass

    def estimate_tid(
        self, altitude_km: float, inclination_deg: float, duration_days: float
    ) -> float:
        """
        Estimate cumulative Total Ionizing Dose (TID).

        Uses a parametric fit for LEO orbits assuming 2.5 mm Aluminum shielding.

        Parameters
        ----------
        altitude_km : float
            Orbit altitude (km).
        inclination_deg : float
            Orbit inclination (deg).
        duration_days : float
            Mission duration (days).

        Returns
        -------
        float
            Estimated TID in kRad(Si).
        """
        # Base rate (400km)
        base_rate = 1.0e-4
        # Altitude scaling (Van Allen belt proxy)
        alt_factor = np.exp((float(altitude_km) - 400.0) / 500.0)
        # Inclination scaling (SAA/Poles)
        inc_factor = 1.0 + 0.5 * np.sin(np.radians(float(inclination_deg)))

        return float(base_rate * alt_factor * inc_factor * duration_days)

    def estimate_seu_rate(self, altitude_km: float, device_cross_section: float = 1.0e-12) -> float:
        """
        Estimate Single Event Upset (SEU) rate from proton flux.

        Parameters
        ----------
        altitude_km : float
            Orbit altitude (km).
        device_cross_section : float, optional
            Device sensitive area ($cm^2/bit$). Default 1e-12.

        Returns
        -------
        float
            Estimated SEUs per bit-day.
        """
        # Proton flux proxy (> 10 MeV)
        flux_p = 100.0 * np.exp((float(altitude_km) - 400.0) / 600.0)
        # 86400 s/day
        rate = flux_p * device_cross_section * 86400.0
        return float(rate)
