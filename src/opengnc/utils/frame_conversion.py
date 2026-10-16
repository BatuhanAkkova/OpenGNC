"""
Frame conversion utilities (ECI, ECEF, LVLH, LLH).
"""

from __future__ import annotations

import numpy as np

from opengnc.utils.euler_utils import rot_z
from opengnc.utils.time_utils import calc_gmst


def eci2ecef(
    r_eci: np.ndarray, v_eci: np.ndarray, jd_ut1: float, dut1: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Convert ECI to ECEF.

    Rotation:
    $\mathbf{r}_{ecef} = \mathbf{R}_z(\theta_{gmst}) \mathbf{r}_{eci}$
    $\mathbf{v}_{ecef} = \mathbf{R}_z(\theta_{gmst}) (\mathbf{v}_{eci} - \mathbf{\omega}_e \times \mathbf{r}_{eci})$

    Parameters
    ----------
    r_eci : np.ndarray
        ECI Position (m).
    v_eci : np.ndarray
        ECI Velocity (m/s).
    jd_ut1 : float
        Julian Date.
    dut1 : float, optional
        UT1-UTC (s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ECEF position vector (m), ECEF velocity vector (m/s)).
    """
    reci = np.asarray(r_eci)
    veci = np.asarray(v_eci)

    gmst = calc_gmst(jd_ut1, dut1)
    # Omega vector of the Earth in ECI
    omega_e = np.array([0.0, 0.0, 7.2921151467e-5])

    r_mat = rot_z(gmst)

    r_ecef = r_mat @ reci
    # v_ecef = R * (v_eci - omega x r_eci)
    v_ecef = r_mat @ (veci - np.cross(omega_e, reci))

    return r_ecef, v_ecef


def ecef2eci(
    r_ecef: np.ndarray, v_ecef: np.ndarray, jd_ut1: float, dut1: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ECEF position and velocity to ECI frame.

    Parameters
    ----------
    r_ecef : np.ndarray
        ECEF position vector (m).
    v_ecef : np.ndarray
        ECEF velocity vector (m/s).
    jd_ut1 : float
        Julian Date (UT1).
    dut1 : float, optional
        UT1-UTC difference (seconds). Default 0.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ECI position vector (m), ECI velocity vector (m/s)).
    """
    recef = np.asarray(r_ecef)
    vecef = np.asarray(v_ecef)

    gmst = calc_gmst(jd_ut1, dut1)
    omega_e = np.array([0.0, 0.0, 7.2921151467e-5])

    r_mat_inv = rot_z(-gmst)

    r_eci = r_mat_inv @ recef
    # v_eci = R_inv * v_ecef + omega x r_eci
    v_eci = r_mat_inv @ vecef + np.cross(omega_e, r_eci)

    return r_eci, v_eci


def eci2lvlh_dcm(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Calculate the DCM from ECI to LVLH (Local Vertical Local Horizontal).

    LVLH Frame (Nadir-Pointed):
    - Z: Nadir (opposite to position vector)
    - Y: Negative Orbit Normal
    - X: Completes the right-handed system (approx velocity direction)

    Parameters
    ----------
    r_eci : np.ndarray
        ECI position vector (m).
    v_eci : np.ndarray
        ECI velocity vector (m/s).

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix $C_{LVLH/ECI}$.
    """
    reci = np.asarray(r_eci)
    veci = np.asarray(v_eci)

    z_u = -reci / np.linalg.norm(reci)
    h_vec = np.cross(reci, veci)
    y_u = -h_vec / np.linalg.norm(h_vec)
    x_u = np.cross(y_u, z_u)

    return np.vstack((x_u, y_u, z_u))


def eci2llh(r_eci: np.ndarray, jd_ut1: float) -> tuple[float, float, float]:
    """
    Convert ECI position to Geodetic LLH (Latitude, Longitude, Height).

    Parameters
    ----------
    r_eci : np.ndarray
        ECI position vector (m).
    jd_ut1 : float
        Julian Date (UT1).

    Returns
    -------
    tuple[float, float, float]
        (Latitude (rad), Longitude (rad), Altitude (m)).
    """
    # WGS84 Parameters
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    rv = np.asarray(r_eci)
    r_ecef, _ = eci2ecef(rv, np.zeros(3), jd_ut1)
    x, y, z = r_ecef

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - e2))
    alt = 0.0

    # Iterative solution for latitude/altitude
    for _ in range(5):
        n_val = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n_val
        lat = np.arctan2(z, p * (1.0 - e2 * (n_val / (n_val + alt))))

    return float(lat), float(lon), float(alt)


def elements2perifocal_dcm(raan: float, inc: float, arg_p: float) -> np.ndarray:
    """
    Calculates the DCM from perifocal (PQW) to ECI.

    Parameters
    ----------
    raan : float
        Right Ascension of the Ascending Node (rad).
    inc : float
        Inclination (rad).
    arg_p : float
        Argument of Perigee (rad).

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix (PQW to ECI).
    """
    c_r, s_r = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc), np.sin(inc)
    c_p, s_p = np.cos(arg_p), np.sin(arg_p)

    r11 = c_r * c_p - s_r * c_i * s_p
    r12 = -c_r * s_p - s_r * c_i * c_p
    r13 = s_r * s_i

    r21 = s_r * c_p + c_r * c_i * s_p
    r22 = -s_r * s_p + c_r * c_i * c_p
    r23 = -c_r * s_i

    r31 = s_i * s_p
    r32 = s_i * c_p
    r33 = c_i

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def eci2geodetic(r_eci: np.ndarray, jd: float) -> tuple[float, float, float]:
    """
    Converts ECI position to Geodetic coordinates.

    Parameters
    ----------
    r_eci : np.ndarray
        ECI position vector (m).
    jd : float
        Julian Date.

    Returns
    -------
    tuple[float, float, float]
        (Longitude (deg), Latitude (deg), Altitude (m)).
    """
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)

    # Latitude
    lat = np.arcsin(z / r)  # radians

    # Longitude (inertial)
    lon_i = np.arctan2(y, x)  # radians

    # GMST Calc
    gmst = calc_gmst(jd)

    lon = lon_i - gmst
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    R_earth = 6378137.0  # meters
    alt = r - R_earth

    return float(np.degrees(lon)), float(np.degrees(lat)), float(alt)


def eci2eme2000(reci: np.ndarray, veci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ECI (J2000) to EME2000.

    In many contexts these are treated as identical. This function treats ECI as
    Earth-Centered Inertial at epoch of date, and EME2000 as fixed at J2000.

    Parameters
    ----------
    reci : np.ndarray
        ECI position vector (m).
    veci : np.ndarray
        ECI velocity vector (m/s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (EME2000 position vector (m), EME2000 velocity vector (m/s)).
    """
    # Treated as identity (no frame rotation between ECI/EME2000 at J2000 epoch)
    return reci, veci


def eme20002eci(reci: np.ndarray, veci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts EME2000 to ECI (J2000).

    Parameters
    ----------
    reci : np.ndarray
        EME2000 position vector (m).
    veci : np.ndarray
        EME2000 velocity vector (m/s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ECI position vector (m), ECI velocity vector (m/s)).
    """
    return reci, veci


def eci2icrf(reci: np.ndarray, veci: np.ndarray, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ECI (J2000) to ICRF.

    Includes simplified Precession and Nutation.

    Parameters
    ----------
    reci : np.ndarray
        ECI position vector (m).
    veci : np.ndarray
        ECI velocity vector (m/s).
    jd : float
        Julian Date.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ICRF position vector (m), ICRF velocity vector (m/s)).
    """
    t = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Simplified Precession (Lieske et al., 1977)
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * (np.pi / 648000.0)
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * (np.pi / 648000.0)
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * (np.pi / 648000.0)

    P = rot_z(-z) @ rot_y_local(theta) @ rot_z(-zeta)

    r_icrf = P @ reci
    v_icrf = P @ veci

    return r_icrf, v_icrf


def icrf2eci(reci: np.ndarray, veci: np.ndarray, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ICRF to ECI (J2000).

    Parameters
    ----------
    reci : np.ndarray
        ICRF position vector (m).
    veci : np.ndarray
        ICRF velocity vector (m/s).
    jd : float
        Julian Date.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ECI position vector (m), ECI velocity vector (m/s)).
    """
    t = (jd - 2451545.0) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * (np.pi / 648000.0)
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * (np.pi / 648000.0)
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * (np.pi / 648000.0)

    P_inv = rot_z(zeta) @ rot_y_local(-theta) @ rot_z(z)

    reci_out = P_inv @ reci
    veci_out = P_inv @ veci

    return reci_out, veci_out


def rot_y_local(angle: float) -> np.ndarray:
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


def llh2ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """
    Converts Geodetic coordinates (LLH) to ECEF.

    Parameters
    ----------
    lat_rad : float
        Latitude in radians.
    lon_rad : float
        Longitude in radians.
    alt_m : float
        Altitude in meters above ellipsoid.

    Returns
    -------
    np.ndarray
        Position in ECEF [m].
    """
    a = 6378137.0  # Semi-major axis [m]
    f = 1.0 / 298.257223563  # Flattening
    e2 = f * (2.0 - f)  # Square of eccentricity

    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1.0 - e2) + alt_m) * np.sin(lat_rad)

    return np.array([x, y, z])
