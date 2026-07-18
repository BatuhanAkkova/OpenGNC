"""
Solar position models based on Julian Date.
"""

import numpy as np


class Sun:
    r"""
    Solar Position Model (Astronomical Almanac).

    Provides the Sun's position vector in the Earth-Centered Inertial (ECI)
    frame (J2000).

    Calculation:
    $\lambda_{ecl} = q + 1.915 \sin(g) + 0.020 \sin(2g)$
    where $q$ is mean longitude and $g$ is mean anomaly.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        """Initialize solar model."""
        pass

    def calculate_sun_eci(self, jd: float) -> np.ndarray:
        """
        Calculate Sun vector in ECI J2000.

        Parameters
        ----------
        jd : float
            Julian Date (UT1).

        Returns
        -------
        np.ndarray
            Sun position $[x, y, z]$ (m).
        """
        # 1. Days since J2000 epoch
        n = float(jd) - 2451545.0

        # 2. Mean anomaly of the Sun (rad)
        g = np.radians(357.529 + 0.98560028 * n)

        # 3. Mean longitude of the Sun (rad)
        q = np.radians(280.459 + 0.98564736 * n)

        # 4. Ecliptic longitude (rad)
        lon_ecl = q + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)

        # 5. Obliquity of the ecliptic (rad)
        eps = np.radians(23.439 - 0.00000036 * n)

        # 6. Physical constant: 1 Astronomical Unit (AU) in meters
        au_m = 149597870700.0

        # 7. Convert from Ecliptic to ECI (Equatorial)
        x = np.cos(lon_ecl)
        y = np.cos(eps) * np.sin(lon_ecl)
        z = np.sin(eps) * np.sin(lon_ecl)

        return np.array([x, y, z]) * au_m




