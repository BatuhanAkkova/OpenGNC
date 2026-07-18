"""
Atmospheric density models (Exponential, Harris-Priester, NRLMSISE-00, JB2008).
"""

from datetime import datetime
from typing import Any

import numpy as np
import pymsis

from opengnc.environment.solar import Sun
from opengnc.utils.frame_conversion import eci2geodetic, eci2llh
from opengnc.utils.time_utils import calc_jd


class Exponential:
    r"""
    Exponential Atmospheric Density.

    Model:
    $\rho = \rho_0 \exp\left(-\frac{h - h_0}{H}\right)$

    Parameters
    ----------
    rho0 : float, optional
        Base density at $h_0$ (kg/m^3).
    h0 : float, optional
        Reference altitude (km).
    h_scale : float, optional
        Scale height $H$ (km).
    """

    def __init__(
        self,
        rho0: float = 1.225,
        h0: float = 0.0,
        h_scale: float | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize exponential model parameters.

        Parameters
        ----------
        rho0 : float, optional
            Reference density at altitude $h_0$ (kg/m^3). Default 1.225 (Sea Level).
        h0 : float, optional
            Reference altitude $h_0$ (km). Default 0.0.
        h_scale : float, optional
            Scale height $H$ (km). Default 8.5.
        **kwargs : Any
            Additional arguments, accepts 'H' as an alias for 'h_scale' for backward compatibility.
        """
        self.rho0 = rho0
        self.h0 = h0

        # Handle backward compatibility for 'H'
        if h_scale is not None:
            self.h_scale = h_scale
        elif "H" in kwargs:
            self.h_scale = kwargs["H"]
        else:
            self.h_scale = 8.5

    def get_density(self, r_eci: np.ndarray, jd: float) -> float:
        """
        Calculate local density.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        jd : float
            Julian Date (UT1).

        Returns
        -------
        float
            Atmospheric density ($kg/m^3$).
        """
        rv = np.asarray(r_eci)
        _, _, h_m = eci2geodetic(rv, jd)

        if h_m < 0:
            return self.rho0

        h_km = h_m / 1000.0
        rho = self.rho0 * np.exp(-(h_km - self.h0) / self.h_scale)
        return float(rho)


class HarrisPriester:
    r"""
    Harris-Priester Diurnal Bulge Model.

    Equation:
    $\rho = \rho_{min} + (\rho_{max} - \rho_{min}) \cos^n(\frac{\psi}{2})$

    Parameters
    ----------
    lag_deg : float, optional
        Solar lag angle (degrees). Default 30.0.
    """

    def __init__(self, lag_deg: float = 30.0) -> None:
        """
        Initialize HP model with solar lag and ephemeris.

        Parameters
        ----------
        lag_deg : float, optional
            Diurnal bulge lag angle behind the sub-solar point (deg).
            Default 30.0.
        """
        self.lag = np.radians(lag_deg)
        self.sun_model = Sun()

    def get_density(self, r_eci: np.ndarray, jd: float) -> float:
        """
        Calculate Harris-Priester interpolated density.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        jd : float
            Julian Date (UT1).

        Returns
        -------
        float
            Density ($kg/m^3$).
        """
        rv = np.asarray(r_eci)
        r_sun = self.sun_model.calculate_sun_eci(jd)

        sun_norm = np.linalg.norm(r_sun)
        sun_u = r_sun / sun_norm if sun_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

        # Apex rotation
        cl, sl = np.cos(self.lag), np.sin(self.lag)
        apex = np.array([
            sun_u[0] * cl - sun_u[1] * sl,
            sun_u[0] * sl + sun_u[1] * cl,
            sun_u[2],
        ])

        r_mag = np.linalg.norm(rv)
        r_u = rv / r_mag if r_mag > 1e-12 else np.array([1.0, 0.0, 0.0])
        cos_psi = np.dot(r_u, apex)

        n_pow = 2
        cos_term = np.abs(np.cos(np.arccos(np.clip(cos_psi, -1.0, 1.0)) / 2.0)) ** n_pow

        # Profile Lookup
        rho_table = [
            (100000.0, 4.974e-07, 4.974e-07), (120000.0, 2.490e-08, 2.490e-08),
            (130000.0, 8.377e-09, 8.710e-09), (140000.0, 3.899e-09, 4.059e-09),
            (150000.0, 2.122e-09, 2.215e-09), (160000.0, 1.263e-09, 1.344e-09),
            (170000.0, 8.008e-10, 8.758e-10), (180000.0, 5.283e-10, 6.010e-10),
            (190000.0, 3.617e-10, 4.297e-10), (200000.0, 2.557e-10, 3.162e-10),
            (210000.0, 1.839e-10, 2.396e-10), (220000.0, 1.341e-10, 1.853e-10),
            (230000.0, 9.949e-11, 1.455e-10), (240000.0, 7.488e-11, 1.157e-10),
            (250000.0, 5.709e-11, 9.308e-11), (260000.0, 4.403e-11, 7.555e-11),
            (270000.0, 3.430e-11, 6.182e-11), (280000.0, 2.697e-11, 5.095e-11),
            (290000.0, 2.139e-11, 4.226e-11), (300000.0, 1.708e-11, 3.526e-11),
            (320000.0, 1.099e-11, 2.511e-11), (340000.0, 7.214e-12, 1.819e-11),
            (360000.0, 4.824e-12, 1.337e-11), (380000.0, 3.274e-12, 9.955e-12),
            (400000.0, 2.249e-12, 7.492e-12), (420000.0, 1.558e-12, 5.684e-12),
            (440000.0, 1.091e-12, 4.355e-12), (460000.0, 7.701e-13, 3.362e-12),
            (480000.0, 5.474e-13, 2.612e-12), (500000.0, 3.916e-13, 2.042e-12),
            (520000.0, 2.819e-13, 1.605e-12), (540000.0, 2.042e-13, 1.267e-12),
            (560000.0, 1.488e-13, 1.005e-12), (580000.0, 1.092e-13, 7.997e-13),
            (600000.0, 8.070e-14, 6.390e-13), (620000.0, 6.012e-14, 5.123e-13),
            (640000.0, 4.519e-14, 4.121e-13), (660000.0, 3.430e-14, 3.325e-13),
            (680000.0, 2.632e-14, 2.691e-13), (700000.0, 2.043e-14, 2.185e-13),
            (720000.0, 1.607e-14, 1.779e-13), (740000.0, 1.281e-14, 1.452e-13),
            (760000.0, 1.036e-14, 1.190e-13), (780000.0, 8.496e-15, 9.776e-14),
            (800000.0, 7.069e-15, 8.059e-14), (840000.0, 4.680e-15, 5.741e-14),
            (880000.0, 3.200e-15, 4.210e-14), (920000.0, 2.210e-15, 3.130e-14),
            (960000.0, 1.560e-15, 2.360e-14), (1000000.0, 1.150e-15, 1.810e-14),
        ]

        _, _, h_m = eci2llh(rv, jd)

        if h_m < rho_table[0][0]:
            return rho_table[0][1]
        if h_m > rho_table[-1][0]:
            return rho_table[-1][1]

        idx = 0
        while idx < len(rho_table) - 2 and h_m > rho_table[idx + 1][0]:
            idx += 1

        h1, rho_min1, rho_max1 = rho_table[idx]
        h2, rho_min2, rho_max2 = rho_table[idx + 1]

        frac = (h_m - h1) / (h2 - h1)
        rho_min = np.exp(np.log(rho_min1) + frac * (np.log(rho_min2) - np.log(rho_min1)))
        rho_max = np.exp(np.log(rho_max1) + frac * (np.log(rho_max2) - np.log(rho_max1)))

        return float(rho_min + (rho_max - rho_min) * cos_term)


class NRLMSISE00:
    """
    NRLMSISE-00 high-fidelity atmospheric density model.

    The standard empirical model of the Earth's atmosphere from ground to
    space. Accounts for solar activity, geomagnetic storms, and seasonal
    variations.

    Notes
    -----
    Requires the `pymsis` package.
    """

    def __init__(self) -> None:
        """Initialize NRLMSISE-00 model."""
        pass

    def get_density(self, r_eci: np.ndarray, date: datetime) -> float:
        """
        Get total mass density using NRLMSISE-00.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        date : datetime
            Current UTC time.

        Returns
        -------
        float
            Atmospheric density ($kg/m^3$).
        """
        rv = np.asarray(r_eci)
        jd, jdfrac = calc_jd(date.year, date.month, date.day, date.hour, date.minute, date.second)
        jdf = jd + jdfrac

        lon_deg, lat_deg, alt_m = eci2geodetic(rv, jdf)

        # alt in km for pymsis
        output = pymsis.calculate(date, lon_deg, lat_deg, alt_m / 1000.0)

        output = np.squeeze(output)
        if output.ndim == 0:
            rho = float(output)
        else:
            rho = float(output[0])

        return rho


class JB2008:
    """
    Simplified Jacchia-Bowman 2008 (JB2008) Atmosphere Model.

    A high-accuracy model based on Jacchia's diffusion equations,
    driven by solar indices (F10.7, S10, M10, etc).

    Parameters
    ----------
    space_weather : Optional[Any], optional
        SpaceWeather model for fetching real-time indices.
    """

    def __init__(self, space_weather: Any | None = None) -> None:
        """Initialize JB2008 with space weather source."""
        from opengnc.environment.space_weather import SpaceWeather

        self.sw = space_weather if space_weather else SpaceWeather()

    def get_density(self, r_eci: np.ndarray, jd: float) -> float:
        """
        Calculate density using solar-scaled thermospheric approximation.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        jd : float
            Julian Date (UT1).

        Returns
        -------
        float
            Atmospheric density ($kg/m^3$).
        """
        rv = np.asarray(r_eci)
        _, _, h_m = eci2geodetic(rv, jd)

        indices = self.sw.get_indices(jd)
        f10 = indices.get("f107", 150.0)
        f10_avg = indices.get("f107_avg", 150.0)

        h_scale = 7.0 + 0.05 * (h_m / 1000.0)
        rho_base = 1.225 * np.exp(-(h_m / 1000.0) / h_scale)

        phi = (f10 + f10_avg) / 2.0
        solar_factor = 1.0 + 0.01 * (phi - 70.0)

        return float(rho_base * solar_factor)


class CIRA72:
    """
    COSPAR International Reference Atmosphere (CIRA) 1972 simplified version.
    """

    def __init__(self) -> None:
        """Initialize simplified CIRA-72 model."""
        pass

    def get_density(self, r_eci: np.ndarray, jd: float) -> float:
        """
        Calculate density using log-polynomial fit.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI position vector (m).
        jd : float
            Julian Date.

        Returns
        -------
        float
            Atmospheric density ($kg/m^3$).
        """
        rv = np.asarray(r_eci)
        _, _, h = eci2geodetic(rv, jd)
        h_km = h / 1000.0

        if h_km < 100:
            return float(1.225 * np.exp(-h_km / 8.5))

        log_rho = -9.0 - 0.015 * (h_km - 100.0) + 1.2e-5 * (h_km - 100.0) ** 2
        return float(10**log_rho)




