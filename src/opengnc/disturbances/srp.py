"""
Solar Radiation Pressure (SRP) acceleration models.
"""

from typing import cast

import numpy as np

from opengnc.environment.solar import Sun


class Canonball:
    r"""
    Solar Radiation Pressure (SRP) 'Cannonball' Model.

    Calculates acceleration due to solar photon momentum transfer, assuming
    a spherical spacecraft with uniform optical properties.

    Acceleration Equation:
    $\mathbf{a}_{srp} = -\nu P_{AU} \left(\frac{AU}{d_{sun}}\right)^2 C_r \frac{A}{m} \hat{\mathbf{u}}_{sun}$

    Parameters
    ----------
    None
        Initialized with default solar pressure at 1 AU.
    """

    def __init__(self) -> None:
        """
        Initialize Sun ephemeris and reference pressure.
        """
        self.sun_model = Sun()
        self.P_sun = 4.56e-6  # N/m^2 (Ref solar pressure at 1 AU)

    def get_acceleration(
        self,
        r_eci: np.ndarray,
        jd: float,
        mass: float,
        area: float,
        cr: float
    ) -> np.ndarray:
        r"""
        Calculate SRP acceleration vector.

        Formula:
        $\mathbf{a}_{srp} = -\nu P_{AU} \left(\frac{AU}{d_{sun}}\right)^2 C_r \frac{A}{m} \hat{\mathbf{u}}_{sun}$

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position (m).
        jd : float
            Julian Date.
        mass : float
            Total mass (kg).
        area : float
            Solar cross-sectional area ($m^2$).
        cr : float
            Radiation pressure coefficient $[1, 2]$.

        Returns
        -------
        np.ndarray
            Acceleration vector ($m/s^2$).
        """
        r_val = np.asarray(r_eci)
        r_sun = self.sun_model.calculate_sun_eci(jd)

        sat_to_sun = r_sun - r_val
        dist_sun = np.linalg.norm(sat_to_sun)
        u_sun = sat_to_sun / dist_sun

        # Shadow factor (nu)
        nu = self.check_eclipse(r_val, r_sun)
        if nu < 1e-6:
            return cast(np.ndarray, np.zeros(3))

        # Pressure scaling (Inverse Square Law)
        au = 149597870700.0
        p_dist = self.P_sun * (au / dist_sun) ** 2

        acc_mag = nu * p_dist * cr * (area / mass)
        return cast(np.ndarray, -acc_mag * u_sun)

    def check_eclipse(self, r_sat: np.ndarray, r_sun: np.ndarray) -> float:
        """
        Determine if the spacecraft is in Earth's shadow (Cylindrical Model).

        Parameters
        ----------
        r_sat : np.ndarray
            Satellite ECI position.
        r_sun : np.ndarray
            Sun ECI position.

        Returns
        -------
        float
            Shadow factor [0=Full shadow, 1=Full sunlight].
        """
        u_sun = r_sun / np.linalg.norm(r_sun)
        s = np.dot(r_sat, u_sun)

        if s > 0:
            return 1.0

        r_perp_sq = np.dot(r_sat, r_sat) - s * s
        R_earth = 6378137.0

        if r_perp_sq < R_earth**2:
            return 0.0
        return 1.0




