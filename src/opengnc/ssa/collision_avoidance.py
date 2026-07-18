"""
Collision Avoidance Maneuver (CAM) Planning.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .conjunction import compute_pc_chan


def plan_collision_avoidance_maneuver(
    r_sat_tca: np.ndarray,
    v_sat_tca: np.ndarray,
    cov_sat_tca: np.ndarray,
    r_deb_tca: np.ndarray,
    v_deb_tca: np.ndarray,
    cov_deb_tca: np.ndarray,
    hbr: float,
    t_man_before_tca: float,
    pc_limit: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """
    Plan an impulsive Collision Avoidance Maneuver (CAM).

    Finds the minimum Delta-V magnitude to reduce Probability of Collision (Pc)
    below the specified limit.

    Assumes Keplerian motion for sensitivity matrix (State Transition Matrix).

    Parameters
    ----------
    r_sat_tca, v_sat_tca : np.ndarray
        ECI state of the satellite at predicted TCA (m, m/s).
    cov_sat_tca : np.ndarray
        Covariance of the satellite at TCA (3x3).
    r_deb_tca, v_deb_tca : np.ndarray
        ECI state of the debris at predicted TCA (m, m/s).
    cov_deb_tca : np.ndarray
        Covariance of the debris at TCA (3x3).
    hbr : float
        Hard Body Radius (m).
    t_man_before_tca : float
        Time before TCA to perform the maneuver (s).
    pc_limit : float
        Maximum allowable Probability of Collision.

    Returns
    -------
    Tuple[np.ndarray, float]
        - Delta-V vector in ECI (m/s).
        - Probability of Collision after maneuver.
    """
    # 1. State Transition Matrix (STM) Approximation (Keplerian)
    # For short durations, we can use the STM Phi_rv (sensitivity to velocity)
    # A simple Hill-Clohessy-Wiltshire (HCW) or Keplerian STM can be used.
    # Here we use a simplified sensitivity: dr_tca = Phi_rv * dv

    mu = 3.986004418e14
    r_mag = np.linalg.norm(r_sat_tca)
    n = np.sqrt(mu / r_mag**3) # Mean motion

    # We want Phi_rv(t_tca, t_man)
    tau = t_man_before_tca

    # Simplified Phi_rv for circular orbits (HCW-based) in ECI
    # This is a heuristic approximation. For high fidelity, use numerical STM.
    # We will assume a radial-track-cross (RTN) frame first.

    def get_cam_pc(dv_rtn: np.ndarray) -> float:
        # Convert dv_rtn to dr_tca_rtn (using HCW Phi_rv)
        # Phi_rv [3x3] for HCW:
        # dr_r = (1/n) * sin(n*t) * dv_r + (2/n) * (1-cos(n*t)) * dv_t
        # dr_t = (2/n) * (cos(n*t)-1) * dv_r + (1/n) * (4*sin(n*t)-3*n*t) * dv_t
        # dr_n = (1/n) * sin(n*t) * dv_n

        dr_r = (1.0/n) * np.sin(n*tau) * dv_rtn[0] + (2.0/n) * (1.0 - np.cos(n*tau)) * dv_rtn[1]
        dr_t = (2.0/n) * (np.cos(n*tau) - 1.0) * dv_rtn[0] + (1.0/n) * (4.0*np.sin(n*tau) - 3.0*n*tau) * dv_rtn[1]
        dr_n = (1.0/n) * np.sin(n*tau) * dv_rtn[2]

        dr_rtn = np.array([dr_r, dr_t, dr_n])

        # Convert RTN displacement to ECI
        # Construct RTN frame at TCA (approximation)
        u_r = r_sat_tca / r_mag
        u_n = np.cross(r_sat_tca, v_sat_tca)
        u_n /= np.linalg.norm(u_n)
        u_t = np.cross(u_n, u_r)
        m_rtn_to_eci = np.vstack([u_r, u_t, u_n]).T

        dr_eci = m_rtn_to_eci @ dr_rtn

        # New satellite position at TCA
        r_sat_new = r_sat_tca + dr_eci

        # Compute Pc
        pc = compute_pc_chan(
            r_sat_new, v_sat_tca, cov_sat_tca,
            r_deb_tca, v_deb_tca, cov_deb_tca,
            hbr
        )
        return pc

    # Optimization objective: minimize magnitude of DV
    # subject to Pc < pc_limit

    res = minimize(
        lambda x: np.linalg.norm(x),
        x0=np.array([0.0, 1e-4, 0.0]), # Small along-track start
        constraints={'type': 'ineq', 'fun': lambda x: pc_limit - get_cam_pc(x)},
        bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)] # Limit to 10cm/s for safety
    )

    dv_rtn_opt = res.x

    # Re-construct RTN to ECI transformation
    u_r = r_sat_tca / r_mag
    u_n = np.cross(r_sat_tca, v_sat_tca)
    u_n /= np.linalg.norm(u_n)
    u_t = np.cross(u_n, u_r)
    m_rtn_to_eci = np.vstack([u_r, u_t, u_n]).T

    dv_eci = m_rtn_to_eci @ dv_rtn_opt
    final_pc = get_cam_pc(dv_rtn_opt)

    return dv_eci, final_pc
