"""
Conversions between ECI position/velocity and Keplerian orbital elements.
"""

import numpy as np


def eci2kepler(
    r_eci: np.ndarray, v_eci: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Convert ECI Cartesian state to Keplerian orbital elements.

    Parameters
    ----------
    r_eci : np.ndarray
        ECI position vector $[x, y, z]$ (m).
    v_eci : np.ndarray
        ECI velocity vector $[v_x, v_y, v_z]$ (m/s).

    Returns
    -------
    tuple
        (a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper)
        - a: Semi-major axis (m)
        - ecc: Eccentricity
        - incl: Inclination (rad)
        - raan: Right Ascension of the Ascending Node (rad)
        - argp: Argument of Perigee (rad)
        - nu: True Anomaly (rad)
        - M: Mean Anomaly (rad)
        - E: Eccentric Anomaly (rad)
        - p: Semi-latus rectum (m)
        - arglat: Argument of Latitude (rad)
        - truelon: True Longitude (rad)
        - lonper: Longitude of Perigee (rad)
    """
    # 1. Earth Gravitational Parameter (m^3/s^2)
    mu = 398600.4415e9

    rv = np.asarray(r_eci)
    vv = np.asarray(v_eci)

    r_mag = float(np.linalg.norm(rv))
    v_mag = float(np.linalg.norm(vv))

    # 2. Angular momentum
    h_vec = np.cross(rv, vv)
    h_mag = float(np.linalg.norm(h_vec))

    # 3. Node vector
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n_mag = float(np.linalg.norm(n_vec))

    # 4. Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * rv - np.dot(rv, vv) * vv) / mu
    ecc = float(np.linalg.norm(e_vec))

    # 5. Semi-major axis and Semi-latus rectum
    energy = float((v_mag**2 / 2.0) - (mu / r_mag))
    if abs(energy) < 1e-12:  # Parabolic
        a = np.inf
    else:
        a = float(-mu / (2.0 * energy))
    p = float(h_mag**2 / mu)

    # 6. Inclination
    incl = float(np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0)))

    # 7. RAAN (Node)
    if n_mag < 1e-12:
        raan = 0.0
    else:
        raan = float(np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0)))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan

    # 8. Argument of Perigee
    if n_mag < 1e-12:
        argp = 0.0  # Placeholder for equatorial orbits
    else:
        if ecc < 1e-12:
            argp = 0.0
        else:
            argp = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * ecc), -1.0, 1.0)))
            if e_vec[2] < 0:
                argp = 2.0 * np.pi - argp

    # 9. True Anomaly
    if ecc < 1e-12:
        nu = 0.0  # Circular orbit placeholder
    else:
        nu = float(np.arccos(np.clip(np.dot(e_vec, rv) / (ecc * r_mag), -1.0, 1.0)))
        if np.dot(rv, vv) < 0:
            nu = 2.0 * np.pi - nu

    # 10. Special Longitudes (Singularities)
    # Argument of Latitude (Circular or Inclined)
    if n_mag < 1e-12:
        arglat = 0.0
    else:
        arglat = float(np.arccos(np.clip(np.dot(n_vec, rv) / (n_mag * r_mag), -1.0, 1.0)))
        if rv[2] < 0:
            arglat = 2.0 * np.pi - arglat

    # Longitude of Perigee (Equatorial Elliptical)
    if ecc < 1e-12:
        lonper = 0.0
    else:
        lonper = float(np.arccos(np.clip(e_vec[0] / ecc, -1.0, 1.0)))
        if e_vec[1] < 0:
            lonper = 2.0 * np.pi - lonper

    # True Longitude (Equatorial Circular)
    truelon = float(np.arccos(np.clip(rv[0] / r_mag, -1.0, 1.0)))
    if rv[1] < 0:
        truelon = 2.0 * np.pi - truelon

    # 11. Mean and Eccentric Anomaly
    e_anom, m_anom = anomalies(ecc, nu)

    return a, ecc, incl, raan, argp, nu, m_anom, e_anom, p, arglat, truelon, lonper


def kepler2eci(
    a: float, ecc: float, incl: float, raan: float, argp: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian elements to ECI Cartesian state.

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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Position vector (m), Velocity vector (m/s)).
    """
    mu = 398600.4415e9
    p = a * (1.0 - ecc**2)

    # 1. Position and Velocity in Perifocal Frame (PQW)
    cos_nu, sin_nu = np.cos(nu), np.sin(nu)
    r_pqw = np.array([cos_nu, sin_nu, 0.0]) * (p / (1.0 + ecc * cos_nu))
    v_pqw = np.array([-sin_nu, ecc + cos_nu, 0.0]) * np.sqrt(mu / p)

    # 2. Rotation Matrix: Perifocal to ECI
    # R = Rz(raan) * Rx(incl) * Rz(argp)
    r_mat = rot_z(raan) @ rot_x(incl) @ rot_z(argp)

    return r_mat @ r_pqw, r_mat @ v_pqw


def anomalies(ecc: float, nu: float) -> tuple[float, float]:
    """
    Compute Eccentric and Mean anomaly from True anomaly.

    Parameters
    ----------
    ecc : float
        Eccentricity.
    nu : float
        True Anomaly (rad).

    Returns
    -------
    tuple[float, float]
        (Eccentric Anomaly, Mean Anomaly) in radians.
    """
    if ecc < 1e-12:  # Circular
        return nu, nu

    if ecc < 1.0:  # Elliptical
        e_anom = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(nu / 2))
        m_anom = e_anom - ecc * np.sin(e_anom)
    elif ecc > 1.0:  # Hyperbolic
        e_anom = 2 * np.arctanh(np.sqrt((ecc - 1) / (ecc + 1)) * np.tan(nu / 2))
        m_anom = ecc * np.sinh(e_anom) - e_anom
    else:  # Parabolic
        e_anom = np.tan(nu / 2)
        m_anom = e_anom + (e_anom**3) / 3.0

    return float(e_anom % (2 * np.pi)), float(m_anom % (2 * np.pi))


def rot_x(angle: float) -> np.ndarray:
    """
    Rotation matrix for rotation about x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    return np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )


def rot_y(angle: float) -> np.ndarray:
    """
    Rotation matrix for rotation about y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )


def rot_z(angle: float) -> np.ndarray:
    """
    Rotation matrix for rotation about z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
