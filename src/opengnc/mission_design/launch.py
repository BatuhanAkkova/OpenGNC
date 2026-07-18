"""
Launch window and injection state computation utilities.
"""

from typing import Any

import numpy as np

from opengnc.utils.frame_conversion import ecef2eci, llh2ecef


def calculate_launch_windows(jd_start: float, jd_end: float, inc_deg: float, raan_deg: float, lat_deg: float, lon_deg: float, step_sec: float = 60) -> list[dict[str, Any]]:
    """
    Calculates launch windows by finding times when the launch site intersects the target orbit plane.

    Args:
        jd_start (float): Start Julian Date.
        jd_end (float): End Julian Date.
        inc_deg (float): Target Inclination [deg].
        raan_deg (float): Target RAAN [deg].
        lat_deg (float): Launch site Latitude [deg].
        lon_deg (float): Launch site Longitude [deg].
        step_sec (float): Search step size [s].

    Returns
    -------
        dict: list of dicts with 'jd', 'type' (Ascending/Descending), and 'azimuth_deg'.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    # Orbit plane normal in ECI
    inc_rad = np.radians(inc_deg)
    raan_rad = np.radians(raan_deg)
    N_orbit = np.array(
        [np.sin(inc_rad) * np.sin(raan_rad), -np.sin(inc_rad) * np.cos(raan_rad), np.cos(inc_rad)]
    )

    # Position in ECEF (Constant)
    r_site_ecef = llh2ecef(lat_rad, lon_rad, 0.0)

    total_sec = (jd_end - jd_start) * 86400.0
    t_array = np.arange(0, total_sec, step_sec)
    jd_array = jd_start + t_array / 86400.0

    dot_products_list: list[float] = []

    for jd in jd_array:
        # Convert to ECI
        r_site_eci, _ = eci2_ecef_or_inverse_wrapper(r_site_ecef, jd)
        dot = np.dot(r_site_eci, N_orbit)
        dot_products_list.append(float(dot))

    dot_products = np.array(dot_products_list, dtype=float)

    # Find sign changes
    windows = []
    for i in range(len(dot_products) - 1):
        if np.sign(dot_products[i]) != np.sign(dot_products[i + 1]):
            # Crosses plane
            jd_cross = jd_array[i]

            # Determine Ascending or Descending
            rate = dot_products[i + 1] - dot_products[i]

            # Azimuth calculation
            cos_phi = np.cos(lat_rad)
            if abs(cos_phi) < 1e-12:
                azimuth_deg = 0.0  # Pole

            else:
                sin_psi = np.cos(inc_rad) / cos_phi
                if abs(sin_psi) <= 1.0:
                    psi_rad = np.arcsin(sin_psi)
                    azimuth_deg = np.degrees(psi_rad)
                else:
                    azimuth_deg = np.nan  # No launching directly

            windows.append({"jd": jd_cross, "rate": rate, "azimuth_approx_deg": azimuth_deg})

    return windows


def eci2_ecef_or_inverse_wrapper(recef: np.ndarray, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Temporary local wrapper to use ecef2eci avoiding import loop or correct usage"""
    return ecef2eci(recef, np.zeros(3), jd)


def compute_injection_state(
    lat_deg: float, lon_deg: float, alt_m: float, azimuth_deg: float, flight_path_angle_deg: float, speed_mps: float, jd: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes ECI state vector at insertion.

    Args:
        lat_deg (float): Latitude [deg].
        lon_deg (float): Longitude [deg].
        alt_m (float): Altitude [m].
        azimuth_deg (float): Azimuth from North [deg].
        flight_path_angle_deg (float): Flight Path Angle from horizontal [deg].
        speed_mps (float): Speed magnitude [m/s].
        jd (float): Julian Date at injection.

    Returns
    -------
        tuple: (r_eci [m], v_eci [m/s])
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    az_rad = np.radians(azimuth_deg)
    fpa_rad = np.radians(flight_path_angle_deg)

    # Position in ECEF
    r_ecef = llh2ecef(lat_rad, lon_rad, alt_m)

    # Velocity in Topocentric ENU
    v_east = speed_mps * np.cos(fpa_rad) * np.sin(az_rad)
    v_north = speed_mps * np.cos(fpa_rad) * np.cos(az_rad)
    v_up = speed_mps * np.sin(fpa_rad)

    v_enu = np.array([v_east, v_north, v_up])

    # Rotation ENU to ECEF
    cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)

    E = np.array([-sin_lon, cos_lon, 0])
    N_vec = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    U = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    R_enu2ecef = np.vstack((E, N_vec, U)).T  # Columns are E, N, U

    v_ecef_rel = R_enu2ecef @ v_enu

    # Add Coriolis (Earth rotation) to get inertial velocity represented in ECEF
    omega_earth = 7.292115e-5  # rad/s
    omega_vec = np.array([0, 0, omega_earth])

    v_ecef_inertial = v_ecef_rel + np.cross(omega_vec, r_ecef)

    # Convert to ECI
    r_eci, v_eci = ecef2eci(r_ecef, v_ecef_inertial, jd)

    return r_eci, v_eci


def calculate_deployment_sequence(
    planes: int, sats_per_plane: int, phasing_parameter_f: int, inc_deg: float, base_raan_deg: float = 0.0, base_ta_deg: float = 0.0
) -> list[dict[str, Any]]:
    """
    Computes target RAAN and True Anomaly for each satellite in a Walker-Delta Constellation (T/P/F).

    Args:
        planes (int): Number of orbital planes (P).
        sats_per_plane (int): Number of satellites per plane (S).
        phasing_parameter_f (int): Phasing parameter (F) between adjacent planes (0 <= F <= P-1).
        inc_deg (float): Inclination of all planes [deg] (not actively changing sequence logic, kept for reference).
        base_raan_deg (float): RAAN of Plane 0 [deg].
        base_ta_deg (float): True anomaly of Satellite 0 in Plane 0 [deg].

    Returns
    -------
        list of dict: Each satellite's parameters including 'plane_id', 'sat_id', 'global_id', 'raan_deg', 'ta_deg'
    """
    total_sats = planes * sats_per_plane

    delta_raan = 360.0 / planes
    delta_ta_in_plane = 360.0 / sats_per_plane
    delta_ta_between_planes = 360.0 * phasing_parameter_f / total_sats

    sequence = []

    global_id = 0
    for p in range(planes):
        raan = (base_raan_deg + p * delta_raan) % 360.0

        for s in range(sats_per_plane):
            ta = (base_ta_deg + s * delta_ta_in_plane + p * delta_ta_between_planes) % 360.0

            sequence.append(
                {"global_id": global_id, "plane_id": p, "sat_id": s, "raan_deg": raan, "ta_deg": ta}
            )
            global_id += 1

    return sequence




