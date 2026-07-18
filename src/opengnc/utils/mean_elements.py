"""
Mean element conversions and secular perturbation rates (e.g., J2 secular theory).
"""

import numpy as np


def osculating2mean(
    a: float,
    ecc: float,
    incl: float,
    raan: float,
    argp: float,
    m_anom: float,
    j2: float = 1.08262668e-3,
    re: float = 6378137.0
) -> tuple[float, float, float, float, float, float]:
    """
    Convert osculating Keplerian elements to mean elements using J2 secular theory.

    This implements the secular mapping (ignoring short/long period terms).
    Mean elements are essential for long-term orbit propagation and mission planning.

    Parameters
    ----------
    a : float
        Osculating semi-major axis (m).
    ecc : float
        Osculating eccentricity.
    incl : float
        Osculating inclination (rad).
    raan : float
        Osculating RAAN (rad).
    argp : float
        Osculating Argument of Perigee (rad).
    m_anom : float
        Osculating Mean Anomaly (rad).
    j2 : float, optional
        J2 perturbation coefficient. Default is Earth.
    re : float, optional
        Planet Equatorial Radius (m). Default is Earth.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Mean elements (a_m, e_m, i_m, raan_m, argp_m, M_m).
    """
    # First-order secular theory keeps a, e, i constant.
    # Periodic terms must be subtracted for a full osculating-to-mean mapping.
    # Current implementation serves as a placeholder for secular-only models.
    return a, ecc, incl, raan, argp, m_anom


def get_j2_secular_rates(
    a: float,
    ecc: float,
    incl: float,
    j2: float = 1.08262668e-3,
    re: float = 6378137.0,
    mu: float = 398600.4415e9
) -> tuple[float, float, float]:
    r"""
    Calculate high-precision secular rates due to J2 oblateness.

    Computes $\dot{\Omega}$, $\dot{\omega}$, and the mean motion correction $\dot{M}$.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    ecc : float
        Eccentricity.
    incl : float
        Inclination (rad).
    j2 : float, optional
        J2 perturbation coefficient.
    re : float, optional
        Equatorial radius (m).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).

    Returns
    -------
    tuple[float, float, float]
        (raan_dot, argp_dot, m_dot) in rad/s.
    """
    p = a * (1.0 - ecc**2)
    n0 = np.sqrt(mu / a**3)

    j2_coeff = 1.5 * n0 * j2 * (re / p)**2

    # 1. RAAN rate (Regression of Nodes)
    raan_dot = -j2_coeff * np.cos(incl)

    # 2. Argument of Perigee rate (Apsidal Rotation)
    argp_dot = 0.5 * j2_coeff * (4.0 - 5.0 * np.sin(incl)**2)

    # 3. Mean Anomaly rate (Mean Motion Correction)
    m_dot = n0 + 0.5 * j2_coeff * np.sqrt(1.0 - ecc**2) * (2.0 - 3.0 * np.sin(incl)**2)

    return float(raan_dot), float(argp_dot), float(m_dot)




