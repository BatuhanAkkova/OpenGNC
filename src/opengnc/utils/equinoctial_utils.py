"""
Equinoctial orbital element operations and conversions.
"""

import numpy as np


def kepler2equinoctial(
    a: float,
    ecc: float,
    incl: float,
    raan: float,
    argp: float,
    m_anom: float
) -> tuple[float, float, float, float, float, float]:
    r"""
    Convert Keplerian elements to Classical Equinoctial elements.

    Equinoctial elements $(a, h, k, p, q, \lambda_M)$ are non-singular
    for zero eccentricity and zero/90-deg inclination.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    ecc : float
        Eccentricity.
    incl : float
        Inclination (rad).
    raan : float
        RAAN (rad).
    argp : float
        Argument of Perigee (rad).
    m_anom : float
        Mean Anomaly (rad).

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Equinoctial elements (a, h, k, p, q, mean_lon).
    """
    h = ecc * np.sin(raan + argp)
    k = ecc * np.cos(raan + argp)
    p = np.tan(incl / 2.0) * np.sin(raan)
    q = np.tan(incl / 2.0) * np.cos(raan)
    mean_lon = raan + argp + m_anom

    return (
        float(a),
        float(h),
        float(k),
        float(p),
        float(q),
        float(mean_lon % (2 * np.pi))
    )


def equinoctial2kepler(
    a: float,
    h: float,
    k: float,
    p: float,
    q: float,
    mean_lon: float
) -> tuple[float, float, float, float, float, float, float]:
    """
    Convert Equinoctial elements back to Keplerian elements.

    Parameters
    ----------
    a, h, k, p, q, mean_lon : float
        Equinoctial elements.

    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        Keplerian (a, ecc, incl, raan, argp, nu, M).
    """
    ecc = np.sqrt(h**2 + k**2)
    incl = 2.0 * np.arctan(np.sqrt(p**2 + q**2))
    raan = np.arctan2(p, q)
    argp = np.arctan2(h, k) - raan

    # 1. Solve Equinoctial Kepler Equation for F (Eccentric Longitude)
    f_lon = solve_kepler_equinoctial(mean_lon, h, k)

    sin_f = np.sin(f_lon)
    cos_f = np.cos(f_lon)
    beta = 1.0 / (1.0 + np.sqrt(1.0 - h**2 - k**2))

    # 2. Coordinates in Equinoctial Frame
    x_eq = a * ((1.0 - h**2 * beta) * cos_f + h * k * beta * sin_f - k)
    y_eq = a * ((1.0 - k**2 * beta) * sin_f + h * k * beta * cos_f - h)

    nu_eq = np.arctan2(y_eq, x_eq)
    nu = nu_eq - (raan + argp)
    m_anom = mean_lon - raan - argp

    return (
        float(a),
        float(ecc),
        float(incl),
        float(raan % (2 * np.pi)),
        float(argp % (2 * np.pi)),
        float(nu % (2 * np.pi)),
        float(m_anom % (2 * np.pi))
    )


def solve_kepler_equinoctial(mean_lon: float, h: float, k: float) -> float:
    r"""
    Iteratively solve the Equinoctial Kepler Equation for $F$.

    $\lambda_M = F + h \cos F - k \sin F$

    Parameters
    ----------
    mean_lon : float
        Mean longitude (rad).
    h, k : float
        Equinoctial elements.

    Returns
    -------
    float
        Eccentric longitude F (rad).
    """
    f_lon = mean_lon
    for _ in range(15):
        res = f_lon + h * np.cos(f_lon) - k * np.sin(f_lon) - mean_lon
        der = 1.0 - h * np.sin(f_lon) - k * np.cos(f_lon)
        f_lon = f_lon - res / der
        if abs(res) < 1e-13:
            break
    return float(f_lon)


def equinoctial2eci(
    a: float,
    h: float,
    k: float,
    p: float,
    q: float,
    mean_lon: float,
    mu: float = 398600.4415e9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct conversion from Equinoctial elements to ECI state.

    Parameters
    ----------
    a, h, k, p, q, mean_lon : float
        Equinoctial elements.
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Position vector (m), Velocity vector (m/s)).
    """
    f_lon = solve_kepler_equinoctial(mean_lon, h, k)

    sin_f, cos_f = np.sin(f_lon), np.cos(f_lon)
    beta = 1.0 / (1.0 + np.sqrt(1.0 - h**2 - k**2))

    x_eq = a * ((1.0 - h**2 * beta) * cos_f + h * k * beta * sin_f - k)
    y_eq = a * ((1.0 - k**2 * beta) * sin_f + h * k * beta * cos_f - h)

    n = np.sqrt(mu / a**3)
    r = a * (1.0 - k * cos_f - h * sin_f)

    x_dot_eq = (n * a**2 / r) * (h * k * beta * cos_f - (1.0 - h**2 * beta) * sin_f)
    y_dot_eq = (n * a**2 / r) * ((1.0 - k**2 * beta) * cos_f - h * k * beta * sin_f)

    # Orientation vectors
    inv_sq = 1.0 / (1.0 + p**2 + q**2)
    f_vec = np.array([1.0 - p**2 + q**2, 2.0 * p * q, -2.0 * p]) * inv_sq
    g_vec = np.array([2.0 * p * q, 1.0 + p**2 - q**2, 2.0 * q]) * inv_sq

    r_eci = x_eq * f_vec + y_eq * g_vec
    v_eci = x_dot_eq * f_vec + y_dot_eq * g_vec

    return r_eci, v_eci




