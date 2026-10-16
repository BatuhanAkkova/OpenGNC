"""
Porkchop plot grid generation for interplanetary transfers.
"""

from collections.abc import Callable

import numpy as np

from ..guidance.rendezvous import solve_lambert


def generate_porkchop_grid(
    departure_dates: np.ndarray | list[float],
    arrival_dates: np.ndarray | list[float],
    r_dep_func: Callable[[float], np.ndarray],
    v_dep_func: Callable[[float], np.ndarray],
    r_arr_func: Callable[[float], np.ndarray],
    v_arr_func: Callable[[float], np.ndarray],
    mu: float = 398600.4418,
) -> dict[str, np.ndarray | list[float]]:
    """
    Generate a grid of C3 and V-infinity values for interplanetary transfers.

    Computes a 'porkchop' plot grid by solving Lambert's problem for every
    combination of departure and arrival dates.

    Parameters
    ----------
    departure_dates : array-like
        List or array of departure times (e.g., seconds from J2000).
    arrival_dates : array-like
        List or array of arrival times (e.g., seconds from J2000).
    r_dep_func : Callable
        Function returning the planet's position vector [3] at departure time t.
    v_dep_func : Callable
        Function returning the planet's velocity vector [3] at departure time t.
    r_arr_func : Callable
        Function returning the destination's position vector [3] at arrival time t.
    v_arr_func : Callable
        Function returning the destination's velocity vector [3] at arrival time t.
    mu : float, optional
        Gravitational parameter of the central body. Default is Earth.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'departure_dates': Copy of departure axes.
        - 'arrival_dates': Copy of arrival axes.
        - 'c3': (N_arr, N_dep) grid of Launch C3 values ($km^2/s^2$).
        - 'v_inf_arr': (N_arr, N_dep) grid of Arrival V-infinity ($km/s$).
        - 'tof': (N_arr, N_dep) grid of Time-of-flight (s).
    """
    n_dep = len(departure_dates)
    n_arr = len(arrival_dates)

    c3_grid = np.full((n_arr, n_dep), np.nan)
    v_inf_arr_grid = np.full((n_arr, n_dep), np.nan)
    tof_grid = np.full((n_arr, n_dep), np.nan)

    for i, t_dep in enumerate(departure_dates):
        r_dep = r_dep_func(t_dep)
        v_dep_planet = v_dep_func(t_dep)

        for j, t_arr in enumerate(arrival_dates):
            dt = float(t_arr - t_dep)
            if dt <= 0:
                continue

            r_arr = r_arr_func(t_arr)
            v_arr_planet = v_arr_func(t_arr)

            try:
                # Solve Lambert problem
                v_dep_sc, v_arr_sc = solve_lambert(r_dep, r_arr, dt, mu=mu)

                # V_infinity at departure
                v_inf_dep = v_dep_sc - v_dep_planet
                c3_val = np.dot(v_inf_dep, v_inf_dep)

                # V_infinity at arrival
                v_inf_arr_val = np.linalg.norm(v_arr_sc - v_arr_planet)

                c3_grid[j, i] = c3_val
                v_inf_arr_grid[j, i] = v_inf_arr_val
                tof_grid[j, i] = dt

            except Exception:
                continue

    return {
        "departure_dates": departure_dates,
        "arrival_dates": arrival_dates,
        "c3": c3_grid,
        "v_inf_arr": v_inf_arr_grid,
        "tof": tof_grid,
    }
