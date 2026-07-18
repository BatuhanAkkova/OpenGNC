"""
Earth magnetic field models (IGRF, WMM, Tilted Dipole).
"""

from datetime import datetime

import numpy as np

try:
    import ppigrf
except ImportError:
    ppigrf = None


def igrf_field(
    lat: float,
    lon: float,
    alt: float,
    time: datetime | float
) -> np.ndarray:
    """
    Get the International Geomagnetic Reference Field (IGRF) vector.

    Parameters
    ----------
    lat : float
        Geodetic latitude (deg).
    lon : float
        Geodetic longitude (deg).
    alt : float
        Altitude above the WGS84 ellipsoid (km).
    time : datetime.datetime or float
        Time for IGRF calculation. Decimal year preferred for float.

    Returns
    -------
    np.ndarray
        Magnetic field vector $[B_{North}, B_{East}, B_{Down}]$ (nT).

    Raises
    ------
    ImportError
        If `ppigrf` dependency is not installed.
    """
    if ppigrf is None:
        raise ImportError("ppigrf not installed")

    # ppigrf expects (lon, lat, alt, time)
    return np.asarray(ppigrf.igrf(lon, lat, alt, time), dtype=float)


def wmm_field(
    lat: float,
    lon: float,
    alt: float,
    date: datetime | float
) -> np.ndarray:
    """
    Get the World Magnetic Model (WMM) field vector.

    Proxied via IGRF in this implementation for GNC simulation balance.

    Parameters
    ----------
    lat : float
        Geodetic latitude (deg).
    lon : float
        Geodetic longitude (deg).
    alt : float
        Altitude (km).
    date : datetime.datetime or float
        Date for model calculation.

    Returns
    -------
    np.ndarray
        Magnetic field vector $[B_{North}, B_{East}, B_{Down}]$ (nT).
    """
    return igrf_field(lat, lon, alt, date)


def tilted_dipole_field(r_ecef: np.ndarray) -> np.ndarray:
    r"""
    Tilted Dipole Geomagnetic Approximation.

    A simplified model useful for fast orbit propagation. Approximates Earth's
    field as a dipole tilted relative to the geographic poles.

    Equation:
    $\mathbf{B} = \frac{\mu_0}{4\pi} \frac{1}{r^3} [3(\mathbf{m} \cdot \hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}]$

    Parameters
    ----------
    r_ecef : np.ndarray
        Position vector in ECEF frame (m).

    Returns
    -------
    np.ndarray
        Magnetic field vector in ECEF frame (T).
    """
    # 1. Earth magnetic constants (Epoch 2020 approx)
    b0 = 3.12e-5  # Equator field strength (T)
    re_m = 6371200.0  # Earth's mean magnetic radius (m)

    # 2. Magnetic Pole location
    mag_lat = np.deg2rad(78.3)
    mag_lon = np.deg2rad(-71.8)

    theta_p = np.pi / 2 - mag_lat
    phi_p = mag_lon

    # 3. Dipole Moment unit vector in ECEF
    moment_u = np.array([
        np.sin(theta_p) * np.cos(phi_p),
        np.sin(theta_p) * np.sin(phi_p),
        np.cos(theta_p)
    ])

    # 4. Strength factor k = B0 * Re^3
    k_mag = b0 * (re_m**3)

    rv = np.asarray(r_ecef)
    r_mag = np.linalg.norm(rv)

    if r_mag < 1.0:
        return np.zeros(3, dtype=float)

    r_unit = rv / r_mag

    # 5. Dipole Field Equation
    b_ecef = (k_mag / (r_mag**3)) * (3.0 * np.dot(moment_u, r_unit) * r_unit - moment_u)

    return np.asarray(-b_ecef, dtype=float)




