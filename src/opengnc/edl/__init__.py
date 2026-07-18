"""
Entry, Descent, and Landing (EDL) dynamics and utilities.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from opengnc.environment.density import Exponential


def ballistic_entry_dynamics(
    t: float,
    state: np.ndarray,
    cd: float,
    area: float,
    mass: float,
    mu: float = 3.986e14,
    r_planet: float = 6371000.0,
    rho_model: Any | None = None,
) -> np.ndarray:
    r"""
    Ballistic Atmospheric Entry Dynamics (3-DOF).

    Calculates the ECI state derivative for a non-lifting entry vehicle
    subject to spherical gravity and aerodynamic drag.

    Parameters
    ----------
    t : float
        Elapsed time (s).
    state : np.ndarray
        ECI State vector $[x, y, z, v_x, v_y, v_z]$ (m, m/s).
    cd : float
        Drag coefficient.
    area : float
        Reference aerodynamic area ($m^2$).
    mass : float
        Vehicle mass (kg).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    r_planet : float, optional
        Planetary reference radius (m).
    rho_model : Optional[Any]
        Atmospheric density model providing `get_density(r, jd)`.

    Returns
    -------
    np.ndarray
        State derivative $[\dot{r}, \dot{v}]$ (m/s, $m/s^2$).
    """
    s = np.asarray(state)
    r_vec, v_vec = s[:3], s[3:]
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    if rho_model is None:
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8500.0)

    rho = rho_model.get_density(r_vec, 0.0)

    # Drag: a_d = -0.5 * rho * v^2 * Cd * A / m * unit(v)
    dynamic_pressure = 0.5 * rho * v_mag**2
    drag_mag = dynamic_pressure * cd * area
    a_drag = -(drag_mag / mass) * (v_vec / v_mag) if v_mag > 1e-6 else np.zeros(3)

    # Gravity
    a_grav = -(mu / r_mag**3) * r_vec

    return cast(np.ndarray, np.concatenate([v_vec, a_grav + a_drag]))


def lifting_entry_dynamics(
    t: float,
    state: np.ndarray,
    cl: float,
    cd: float,
    bank_angle: float,
    area: float,
    mass: float,
    mu: float = 3.986e14,
    r_planet: float = 6371000.0,
    rho_model: Any | None = None,
) -> np.ndarray:
    """
    Lifting Atmospheric Entry Dynamics with Bank Angle Modulation.

    Calculates the ECI state derivative for a vehicle with non-zero
    Lift-over-Drag (L/D) ratios.

    Parameters
    ----------
    t : float
        Elapsed time (s).
    state : np.ndarray
        ECI State vector (m, m/s).
    cl, cd : float
        Lift and Drag coefficients.
    bank_angle : float
        Rotation of the lift vector about the velocity vector (rad).
    area : float
        Reference area ($m^2$).
    mass : float
        Mass (kg).
    mu, r_planet : float
        Planetary parameters.
    rho_model : Optional[Any]
        Density model.

    Returns
    -------
    np.ndarray
        State derivative.
    """
    s = np.asarray(state)
    r_vec, v_vec = s[:3], s[3:]
    r_mag, v_mag = np.linalg.norm(r_vec), np.linalg.norm(v_vec)

    if rho_model is None:
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8500.0)

    rho = rho_model.get_density(r_vec, 0.0)
    dynamic_pressure = 0.5 * rho * v_mag**2

    # Coordinate system for forces
    u_v = v_vec / v_mag if v_mag > 1e-6 else np.zeros(3)
    u_h = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(u_h)
    u_h = u_h / h_mag if h_mag > 1e-6 else np.zeros(3)
    u_l_v = np.cross(u_v, u_h)  # Lift unit vector in the vertical plane

    # Force magnitudes
    lift_mag = dynamic_pressure * cl * area
    drag_mag = dynamic_pressure * cd * area

    a_drag = -(drag_mag / mass) * u_v
    # Lift components modulated by bank angle
    a_lift = (lift_mag / mass) * (np.cos(bank_angle) * u_l_v + np.sin(bank_angle) * u_h)

    # Gravity
    a_grav = -(mu / r_mag**3) * r_vec

    return cast(np.ndarray, np.concatenate([v_vec, a_grav + a_drag + a_lift]))


def sutton_grave_heating(rho: float, v: float, rn: float) -> float:
    r"""
    Stagnation Point Heat Flux via Sutton-Grave Correlation.

    Estimates the convective heat transfer at the vehicle nose during
    hypersonic atmospheric entry.
    Equation: $\dot{q} = k \sqrt{\rho / r_n} v^3$.

    Parameters
    ----------
    rho : float
        Atmospheric density ($kg/m^3$).
    v : float
        Relative velocity (m/s).
    rn : float
        Nose/Stagnation region radius (m).

    Returns
    -------
    float
        Stagnation heat flux ($W/m^2$).
    """
    k = 1.74153e-4  # Constant for Earth atmospheric species
    return float(k * np.sqrt(rho / rn) * v**3)


def calculate_g_load(acc_vec: np.ndarray) -> float:
    """
    Calculate instantaneous G-load.

    Parameters
    ----------
    acc_vec : np.ndarray
        Net non-gravitational acceleration vector ($m/s^2$).

    Returns
    -------
    float
        Load in Earth g-units.
    """
    g0 = 9.80665
    return float(np.linalg.norm(acc_vec) / g0)


def aerocapture_guidance(
    state: np.ndarray,
    target_apoapsis: float,
    cd: float,
    area: float,
    mass: float,
    planet_params: dict[str, float],
    rho_model: Any,
    cl: float = 0.0
) -> float:
    """
    Predictive-Corrector Aerocapture Guidance.

    Determines the required bank angle to achieve a target exit apoapsis
    by numerically integrating internal trajectories.

    Parameters
    ----------
    state : np.ndarray
        Current spacecraft state $[r, v]$.
    target_apoapsis : float
        Desired apoapsis altitude after atmospheric exit (m).
    cd, area, mass : float
        Vehicle ballistic parameters.
    planet_params : Dict[str, float]
        Dictionary containing 'mu' and 'r_planet'.
    rho_model : Any
        Atmospheric density model.
    cl : float, optional
        Lift coefficient. Defaults to 0 (ballistic).

    Returns
    -------
    float
        Optimized bank angle (rad).
    """
    mu = planet_params.get("mu", 3.986e14)
    r_planet = planet_params.get("r_planet", 6371000.0)
    atm_int = r_planet + 120000.0

    from scipy.integrate import solve_ivp

    def get_exit_apoapsis(s: np.ndarray) -> float:
        rv, vv = s[:3], s[3:]
        r, v = np.linalg.norm(rv), np.linalg.norm(vv)
        energy = 0.5 * v**2 - mu / r
        if energy >= 0:
            return float(np.inf)
        a = -mu / (2 * energy)
        val = 1.0 - (np.linalg.norm(np.cross(rv, vv)) ** 2) / (a * mu)
        e = float(np.sqrt(max(0.0, float(val))))
        return float(a * (1 + e) - r_planet)

    if cl <= 0.0:
        return 0.0

    def predict(bank: float) -> float:
        def dydt(t: float, y: np.ndarray) -> np.ndarray:
            return lifting_entry_dynamics(t, y, cl, cd, bank, area, mass, mu, r_planet, rho_model)

        def exit_check(t: float, y: np.ndarray) -> float:
            return float(np.linalg.norm(y[:3]) - atm_int)
        exit_check.terminal = True  # type: ignore[attr-defined]
        exit_check.direction = 1  # type: ignore[attr-defined]

        sol = solve_ivp(dydt, (0, 3600.0), state, events=exit_check, rtol=1e-4)
        return float(get_exit_apoapsis(sol.y[:, -1]))

    # Simple Bisection
    b_min, b_max = 0.0, np.pi
    for _ in range(8):
        b_mid = (b_min + b_max) / 2
        ap = predict(b_mid)
        if ap > target_apoapsis:
            b_min = b_mid
        else:
            b_max = b_mid

    return float((b_min + b_max) / 2)


def hazard_avoidance(
    r: np.ndarray,
    v: np.ndarray,
    hazards: list[np.ndarray],
    safety_margin: float = 50.0
) -> np.ndarray:
    r"""
    Reactive Hazard Avoidance Maneuver logic.

    Parameters
    ----------
    r, v : np.ndarray
        Current spacecraft landing state.
    hazards : List[np.ndarray]
        Coordinates of detected obstacles.
    safety_margin : float, optional
        Minimum separation distance (m).

    Returns
    -------
    np.ndarray
        Correction $\Delta V$ vector (m/s).
    """
    pos = np.asarray(r)
    for h in hazards:
        h_pos = np.asarray(h)
        dist = np.linalg.norm(pos - h_pos)
        if dist < safety_margin:
            u_div = (pos - h_pos) / max(1e-3, dist)
            return cast(np.ndarray, u_div * 5.0)
    return cast(np.ndarray, np.zeros(3))




