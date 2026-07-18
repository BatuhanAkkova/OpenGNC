"""
Simplified Lunar Ephemeris Model for position calculation.
"""

import numpy as np


class Moon:
    """
    Lunar Ephemeris Model (Vallado 3.5).

    Provides the Moon's position vector in the Earth-Centered Inertial (ECI)
    frame.

    Calculation (Fundamental Arguments):
    $L$ = Mean longitude, $M'$ = Moon mean anomaly, $M$ = Sun mean anomaly,
    $D$ = Mean elongation, $u$ = Mean latitude.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        """Initialize lunar constants."""
        self.mu_moon = 4902.800066e9  # Lunar gravitational parameter (m^3/s^2)

    def calculate_moon_eci(self, jd: float) -> np.ndarray:
        """
        Calculate Moon vector in ECI.

        Parameters
        ----------
        jd : float
            Julian Date (UT1).

        Returns
        -------
        np.ndarray
            Moon position vector $[x, y, z]$ (m).
        """
        # 1. Julian centuries since J2000
        t_cen = (float(jd) - 2451545.0) / 36525.0

        # 2. Mean fundamental arguments [deg]
        lam_m = 218.316 + 481267.8813 * t_cen  # Mean longitude
        m_m = np.radians((134.963 + 477198.8676 * t_cen) % 360)    # Moon mean anomaly
        d_rad = np.radians((297.850 + 445267.1115 * t_cen) % 360)  # Moon mean elongation
        u_rad = np.radians((93.272 + 483202.0175 * t_cen) % 360)   # Moon mean latitude

        # 4. Longitude series (fundamental terms)
        lon = (
            lam_m
            + 6.289 * np.sin(m_m)
            - 1.274 * np.sin(m_m - 2 * d_rad)
            + 0.658 * np.sin(2 * d_rad)
            + 0.214 * np.sin(2 * m_m)
        )
        # Latitude series
        lat = (
            5.128 * np.sin(u_rad)
            + 0.280 * np.sin(u_rad + m_m)
            + 0.277 * np.sin(m_m - u_rad)
            + 0.173 * np.sin(u_rad - 2 * d_rad)
        )

        # 5. Parallax (determines distance)
        hp = (
            0.9508
            + 0.0518 * np.cos(m_m)
            + 0.0095 * np.cos(m_m - 2 * d_rad)
            + 0.0078 * np.cos(2 * d_rad)
            + 0.0028 * np.cos(2 * m_m)
        )

        # 6. Radial distance in km (Mean Earth Radius R_e / sin(hp))
        r_mag_km = 6378.137 / np.sin(np.radians(hp))

        lon_rad = np.radians(lon % 360)
        lat_rad = np.radians(lat % 360)

        # 7. Convert Ecliptic to ECI (Obliquity eps ~ 23.439 deg)
        eps = np.radians(23.439291)  # Obliquity

        x = r_mag_km * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_mag_km * (np.cos(eps) * np.cos(lat_rad) * np.sin(lon_rad) - np.sin(eps) * np.sin(lat_rad))
        z = r_mag_km * (np.sin(eps) * np.cos(lat_rad) * np.sin(lon_rad) + np.cos(eps) * np.sin(lat_rad))

        return np.array([x, y, z]) * 1000.0  # Meters




