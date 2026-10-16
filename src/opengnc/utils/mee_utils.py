"""
Modified Equinoctial Elements (MEE) kinematics and conversions.
"""

import numpy as np


def kepler2mee(
    a: float,
    ecc: float,
    incl: float,
    raan: float,
    argp: float,
    nu: float,
    mu: float = 398600.4415e9,
) -> tuple[float, float, float, float, float, float]:
    """
    Convert Keplerian elements to Modified Equinoctial Elements (MEE).

    MEE are non-singular for circular and equatorial orbits.
    State vector: $[p, f, g, h, k, L]^T$.

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
    nu : float
        True Anomaly (rad).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default is Earth.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        MEE elements (p, f, g, h, k, L).
    """
    p_mee = a * (1.0 - ecc**2)
    f = ecc * np.cos(raan + argp)
    g = ecc * np.sin(raan + argp)
    h = np.tan(incl / 2.0) * np.cos(raan)
    k = np.tan(incl / 2.0) * np.sin(raan)
    l_true = raan + argp + nu

    return (float(p_mee), float(f), float(g), float(h), float(k), float(l_true % (2 * np.pi)))


def mee2eci(
    p: float, f: float, g: float, h: float, k: float, l_true: float, mu: float = 398600.4415e9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Modified Equinoctial Elements to ECI Cartesian state.

    Parameters
    ----------
    p, f, g, h, k, l_true : float
        Modified Equinoctial Elements.
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default is Earth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Position vector (m), Velocity vector (m/s)).
    """
    sin_l = np.sin(l_true)
    cos_l = np.cos(l_true)

    w = 1.0 + f * cos_l + g * sin_l
    r = p / w

    s2 = 1.0 + h**2 + k**2

    r_eci = (r / s2) * np.array(
        [
            cos_l * (1 + h**2 - k**2) + 2 * h * k * sin_l,
            sin_l * (1 - h**2 + k**2) + 2 * h * k * cos_l,
            2 * (h * sin_l - k * cos_l),
        ]
    )

    v_coeff = np.sqrt(mu / p) / s2
    v_eci = v_coeff * np.array(
        [
            -(
                sin_l * (1 + h**2 - k**2)
                - 2 * h * k * cos_l
                + g
                + g * h**2
                - g * k**2
                - 2 * f * h * k
            ),
            (
                cos_l * (1 - h**2 + k**2)
                - 2 * h * k * sin_l
                + f
                - f * h**2
                + f * k**2
                - 2 * g * h * k
            ),
            2 * (cos_l + f) * h + 2 * (sin_l + g) * k,
        ]
    )

    return r_eci, v_eci


def eci2mee(
    r_eci: np.ndarray, v_eci: np.ndarray, mu: float = 398600.4415e9
) -> tuple[float, float, float, float, float, float]:
    """
    Convert ECI Cartesian state to Modified Equinoctial Elements.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector (m).
    v_eci : np.ndarray
        Velocity vector (m/s).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default is Earth.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        MEE elements (p, f, g, h, k, L).
    """
    from .state_to_elements import eci2kepler

    rv = np.asarray(r_eci)
    vv = np.asarray(v_eci)

    a, ecc, incl, raan, argp, nu, _, _, _, _, _, _ = eci2kepler(rv, vv)

    return kepler2mee(a, ecc, incl, raan, argp, nu, mu)
