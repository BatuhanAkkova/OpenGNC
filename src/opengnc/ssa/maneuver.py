"""
Debris Avoidance Maneuver Planning.
"""


from typing import cast

import numpy as np


def plan_avoidance_maneuver(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    r_debris: np.ndarray,
    v_debris: np.ndarray,
    safety_radius: float,
    t_encounter: float,
) -> tuple[np.ndarray, float]:
    """
    Plan an impulsive Debris Avoidance Maneuver (DAM).

    Primarily calculates an along-track thrust to achieve a target miss
    distance at encounter via phasing.

    Parameters
    ----------
    r_sat, v_sat : np.ndarray
        Spacecraft ECI state at planning epoch (m, m/s).
    r_debris, v_debris : np.ndarray
        Debris ECI state at planning epoch (m, m/s).
    safety_radius : float
        Desired minimum separation distance (m).
    t_encounter : float
        Time until predicted conjunction (s).

    Returns
    -------
    Tuple[np.ndarray, float]
        - Delta-V vector in ECI (m/s).
        - Estimated miss distance after maneuver (m).
    """
    rs, vs = np.asarray(r_sat), np.asarray(v_sat)
    rd = np.asarray(r_debris)

    v_mag = float(np.linalg.norm(vs))
    if v_mag < 1e-6:
        raise ValueError("Velocity is too small.")

    # Unit along-track vector
    t_hat = vs / v_mag

    # Current predicted miss (Euclidean distance at encounter/epoch)
    r_rel = rs - rd
    d_curr = float(np.linalg.norm(r_rel))

    if d_curr >= safety_radius:
        return np.zeros(3), float(d_curr)

    # Phasing approximation: d_r_track ~ 3 * dt * dv_t
    d_req = safety_radius - d_curr

    if t_encounter < 10.0:
        # Radial/Direct avoidance (Heuristic)
        dv_mag = d_req / t_encounter
    else:
        # Efficient along-track phasing
        dv_mag = d_req / (3.0 * t_encounter)

    dv_vec = cast(np.ndarray, t_hat * dv_mag)
    d_est = d_curr + np.abs(dv_mag) * (3.0 * t_encounter if t_encounter > 10.0 else t_encounter)

    return dv_vec, float(d_est)




