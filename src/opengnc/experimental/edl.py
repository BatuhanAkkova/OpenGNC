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
    """
    s = np.asarray(state)
    r_vec, v_vec = s[:3], s[3:]
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    if rho_model is None:
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8500.0)

    rho = rho_model.get_density(r_vec, 0.0)

    dynamic_pressure = 0.5 * rho * v_mag**2
    drag_mag = dynamic_pressure * cd * area
    a_drag = -(drag_mag / mass) * (v_vec / v_mag) if v_mag > 1e-6 else np.zeros(3)
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
    """
    s = np.asarray(state)
    r_vec, v_vec = s[:3], s[3:]
    r_mag, v_mag = np.linalg.norm(r_vec), np.linalg.norm(v_vec)

    if rho_model is None:
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8500.0)

    rho = rho_model.get_density(r_vec, 0.0)
    dynamic_pressure = 0.5 * rho * v_mag**2

    u_v = v_vec / v_mag if v_mag > 1e-6 else np.zeros(3)
    u_h = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(u_h)
    u_h = u_h / h_mag if h_mag > 1e-6 else np.zeros(3)
    u_l_v = np.cross(u_v, u_h)

    lift_mag = dynamic_pressure * cl * area
    drag_mag = dynamic_pressure * cd * area

    a_drag = -(drag_mag / mass) * u_v
    a_lift = (lift_mag / mass) * (np.cos(bank_angle) * u_l_v + np.sin(bank_angle) * u_h)
    a_grav = -(mu / r_mag**3) * r_vec

    return cast(np.ndarray, np.concatenate([v_vec, a_grav + a_drag + a_lift]))


def sutton_grave_heating(rho: float, v: float, rn: float) -> float:
    r"""
    Stagnation point heat flux via Sutton-Grave correlation.
    """
    k = 1.74153e-4
    return float(k * np.sqrt(rho / rn) * v**3)


def calculate_g_load(acc_vec: np.ndarray) -> float:
    """
    Calculate instantaneous G-load.
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
    cl: float = 0.0,
) -> float:
    """
    Predictive-corrector aerocapture guidance.
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
    safety_margin: float = 50.0,
) -> np.ndarray:
    r"""
    Reactive hazard avoidance maneuver logic.
    """
    pos = np.asarray(r)
    for h in hazards:
        h_pos = np.asarray(h)
        dist = np.linalg.norm(pos - h_pos)
        if dist < safety_margin:
            u_div = (pos - h_pos) / max(1e-3, dist)
            return cast(np.ndarray, u_div * 5.0)
    return cast(np.ndarray, np.zeros(3))


__all__ = [
    "aerocapture_guidance",
    "ballistic_entry_dynamics",
    "calculate_g_load",
    "hazard_avoidance",
    "lifting_entry_dynamics",
    "sutton_grave_heating",
]
