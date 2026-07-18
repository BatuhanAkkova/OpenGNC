"""
Ground station access and lighting conditions analysis utilities.
"""

from typing import Any

import numpy as np

from opengnc.environment.solar import Sun
from opengnc.utils.frame_conversion import eci2ecef, eci2llh, llh2ecef


def calculate_access_windows(
    t_array: np.ndarray, r_eci_array: np.ndarray, gs_lat_deg: float, gs_lon_deg: float, gs_alt_m: float, min_elevation_deg: float = 5.0, jdut1: float = 2451545.0
) -> dict[str, Any]:
    """
    Calculates access windows (visibility) from a Ground Station for a given trajectory.

    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        gs_lat_deg (float): Ground Station Latitude [deg].
        gs_lon_deg (float): Ground Station Longitude [deg].
        gs_alt_m (float): Ground Station Altitude [m].
        min_elevation_deg (float): Minimum elevation for visibility [deg].
        jdut1 (float): Julian Date UT1 at t=0.

    Returns
    -------
        dict: Containing 'visible_intervals' (list of dicts), 'elevation_history' (degrees).
    """
    gs_lat_rad = np.radians(gs_lat_deg)
    gs_lon_rad = np.radians(gs_lon_deg)
    r_gs_ecef = llh2ecef(gs_lat_rad, gs_lon_rad, gs_alt_m)

    # Zenith vector at GS (Normal to ellipsoid)
    U = np.array(
        [
            np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
            np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
            np.sin(gs_lat_rad),
        ]
    )

    elevations = []
    visibility = []

    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]

        # Convert to ECEF
        r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd_current)

        # Relative vector
        rho_ecef = r_ecef - r_gs_ecef
        rho_norm = np.linalg.norm(rho_ecef)

        if rho_norm == 0:
            elevations.append(90.0)
            visibility.append(True)
            continue

        # Elevation
        elevation_rad = np.arcsin(np.dot(rho_ecef, U) / rho_norm)
        elevation_deg = np.degrees(elevation_rad)

        elevations.append(elevation_deg)
        visibility.append(elevation_deg >= min_elevation_deg)

    # Find intervals
    visible_intervals = []
    in_pass = False
    start_time = None

    for i, is_visible in enumerate(visibility):
        if is_visible and not in_pass:
            in_pass = True
            start_time = t_array[i]
        elif not is_visible and in_pass:
            in_pass = False
            visible_intervals.append(
                {
                    "start_time": start_time,
                    "end_time": t_array[i],
                    "duration": t_array[i] - start_time,
                }
            )

    if in_pass:
        visible_intervals.append(
            {
                "start_time": start_time,
                "end_time": t_array[-1],
                "duration": t_array[-1] - start_time,
            }
        )

    return {"visible_intervals": visible_intervals, "elevation_history": np.array(elevations)}


def calculate_ground_track(t_array: np.ndarray, r_eci_array: np.ndarray, jdut1: float = 2451545.0) -> dict[str, np.ndarray]:
    """
    Calculates ground track coordinates (Lat, Lon, Alt) over time.

    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        jdut1 (float): Julian Date UT1 at t=0.

    Returns
    -------
        dict: Containing 'lat_deg', 'lon_deg', 'alt_m' arrays.
    """
    lats = []
    lons = []
    alts = []

    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]

        lat_rad, lon_rad, alt = eci2llh(r_eci, jd_current)

        # Normalize lon to [-180, 180]
        lon_deg = np.degrees(lon_rad)
        if lon_deg > 180:
            lon_deg -= 360
        elif lon_deg < -180:
            lon_deg += 360

        lats.append(np.degrees(lat_rad))
        lons.append(lon_deg)
        alts.append(alt)

    return {"lat_deg": np.array(lats), "lon_deg": np.array(lons), "alt_m": np.array(alts)}


def calculate_lighting_conditions(t_array: np.ndarray, r_eci_array: np.ndarray, v_eci_array: np.ndarray, jdut1: float = 2451545.0) -> dict[str, np.ndarray]:
    """
    Calculates Beta angle and Eclipse status over time.

    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        v_eci_array (np.ndarray): Velocity history in ECI [m/s], shape (N, 3).
        jdut1 (float): Julian Date UT1 at t=0.

    Returns
    -------
        dict: Containing 'beta_angle_deg', 'eclipse_state' (1 for Sun, 0 for shade).
    """
    sun_model = Sun()

    beta_angles = []
    eclipse_state = []

    R_earth = 6378137.0


    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]
        v_eci = v_eci_array[i]

        # Sun position
        r_sun = sun_model.calculate_sun_eci(jd_current)
        r_sun_norm = np.linalg.norm(r_sun)
        u_sun = r_sun / r_sun_norm

        # Orbit Normal
        h = np.cross(r_eci, v_eci)
        h_norm = np.linalg.norm(h)
        if h_norm == 0:
            beta_angles.append(0.0)
        else:
            u_h = h / h_norm
            # Beta angle = asin(dot(h, u_sun))
            beta_rad = np.arcsin(np.dot(u_h, u_sun))
            beta_angles.append(np.degrees(beta_rad))

        # Eclipse Check (Cylindrical model)
        s = np.dot(r_eci, u_sun)

        if s > 0:
            eclipse_state.append(1.0)  # Sunlight
        else:
            r_perp_sq = np.dot(r_eci, r_eci) - s * s
            if r_perp_sq < R_earth**2:
                eclipse_state.append(0.0)  # Eclipse
            else:
                eclipse_state.append(1.0)  # Sunlight

    return {"beta_angle_deg": np.array(beta_angles), "eclipse_state": np.array(eclipse_state)}


def calculate_constellation_coverage(
    t_array: np.ndarray, r_eci_array_list: list[np.ndarray], target_points_llh: np.ndarray, min_elevation_deg: float = 5.0, jdut1: float = 2451545.0
) -> list[dict[str, float]]:
    """
    Evaluates coverage statistics (gap, revisit time) for a constellation over a list of ground targets.

    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array_list (list of np.ndarray): List of position histories in ECI [m], each shape (N, 3).
        target_points_llh (np.ndarray): Target locations as (M, 2) or (M, 3) matrix [Lat(deg), Lon(deg), Alt(m)].
                                        Altitude defaults to 0 if only 2 columns provided.
        min_elevation_deg (float): Minimum elevation for visibility [deg]. Defaults to 5.0.
        jdut1 (float): Julian Date UT1 at t=0. Defaults to 2451545.0.

    Returns
    -------
        list of dict: For each target point, statistics including 'max_revisit_time_gap',
                      'mean_revisit_time', and 'total_coverage_time'.
    """
    M = len(target_points_llh)
    N = len(t_array)
    num_sats = len(r_eci_array_list)

    # Store results per point
    results = []

    for m in range(M):
        lat_deg = target_points_llh[m, 0]
        lon_deg = target_points_llh[m, 1]
        alt_m = target_points_llh[m, 2] if target_points_llh.shape[1] > 2 else 0.0

        gs_lat_rad = np.radians(lat_deg)
        gs_lon_rad = np.radians(lon_deg)
        r_gs_ecef = llh2ecef(gs_lat_rad, gs_lon_rad, alt_m)

        # Zenith vector at target
        U = np.array(
            [
                np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
                np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
                np.sin(gs_lat_rad),
            ]
        )

        # visibility combined across the constellation
        # visibility[i] = True if ANY satellite is visible at time t_array[i]
        point_visibility = np.zeros(N, dtype=bool)

        for k in range(num_sats):
            r_eci_array = r_eci_array_list[k]

            for i, t in enumerate(t_array):
                if point_visibility[i]:
                    continue  # Already covered by someone else

                jd_current = jdut1 + t / 86400.0
                r_eci = r_eci_array[i]

                # Convert this satellite to ECEF
                r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd_current)

                rho_ecef = r_ecef - r_gs_ecef
                rho_norm = np.linalg.norm(rho_ecef)

                if rho_norm == 0:
                    point_visibility[i] = True
                    continue

                elevation_rad = np.arcsin(np.clip(np.dot(rho_ecef, U) / rho_norm, -1.0, 1.0))
                if np.degrees(elevation_rad) >= min_elevation_deg:
                    point_visibility[i] = True

        # Now compute statistics: coverage gaps and revisit times
        # find contiguous False regions (gaps)
        gaps_list: list[float] = []
        covered_time = 0.0

        in_gap = not point_visibility[0]
        gap_start_t = t_array[0] if in_gap else None

        for i in range(1, N):
            was_visible = point_visibility[i - 1]
            is_visible = point_visibility[i]

            if is_visible:
                covered_time += t_array[i] - t_array[i - 1]

            if not was_visible and is_visible:
                # End of a gap
                gap_duration = t_array[i] - gap_start_t
                gaps_list.append(float(gap_duration))
                in_gap = False
            elif was_visible and not is_visible:
                # Start of a gap
                in_gap = True
                gap_start_t = t_array[i - 1]  # previous time was the last moment it was visible

        if in_gap:
            # gap persists to the end of the simulation
            gaps_list.append(float(t_array[-1] - gap_start_t))

        gaps = np.array(gaps_list, dtype=float)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
        mean_gap = np.mean(gaps) if len(gaps) > 0 else 0.0

        results.append(
            {
                "max_revisit_time_gap": max_gap,
                "mean_revisit_time": mean_gap,
                "total_coverage_time": covered_time,
            }
        )

    return results




