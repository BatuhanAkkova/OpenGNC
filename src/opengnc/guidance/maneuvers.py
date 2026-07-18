"""
Orbital maneuver calculations (Hohmann, Bi-elliptic, Phasing, Plane Change).
"""

from __future__ import annotations

import numpy as np


def hohmann_transfer(
    r1: float, r2: float, mu: float = 398600.4418
) -> tuple[float, float, float]:
    r"""
    Calculate Hohmann transfer $\Delta V$ and time of flight.

    Equations:
    - $\Delta V_1 = \sqrt{\frac{\mu}{r_1}} \left( \sqrt{\frac{2r_2}{r_1+r_2}} - 1 \right)$
    - $\Delta V_2 = \sqrt{\frac{\mu}{r_2}} \left( 1 - \sqrt{\frac{2r_1}{r_1+r_2}} \right)$
    - $t_{trans} = \pi \sqrt{\frac{(r_1+r_2)^3}{8\mu}}$

    Parameters
    ----------
    r1 : float
        Initial circular radius (km).
    r2 : float
        Final circular radius (km).
    mu : float, optional
        Gravitational parameter ($km^3/s^2$). Default Earth.

    Returns
    -------
    tuple[float, float, float]
        (dv1, dv2, time_of_flight) in km/s and s.
    """
    r1_arr = np.asarray(r1, dtype=float)
    r2_arr = np.asarray(r2, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)

    # Semi-major axis of the transfer ellipse
    a_trans = (r1_arr + r2_arr) / 2.0

    # Velocity at periapsis and apoapsis of transfer orbit
    v_trans_p = np.sqrt(mu_arr * (2 / r1_arr - 1 / a_trans))
    v_trans_a = np.sqrt(mu_arr * (2 / r2_arr - 1 / a_trans))

    # Velocity of initial and final circular orbits
    v_c1 = np.sqrt(mu_arr / r1_arr)
    v_c2 = np.sqrt(mu_arr / r2_arr)

    # Calculate Delta-Vs
    dv1 = abs(v_trans_p - v_c1)
    dv2 = abs(v_c2 - v_trans_a)

    # Transfer time (half period)
    t_flight = np.pi * np.sqrt(a_trans**3 / mu_arr)

    return float(dv1), float(dv2), float(t_flight)


def bi_elliptic_transfer(
    r1: float, r2: float, rb: float, mu: float = 398600.4418
) -> tuple[float, float, float, float]:
    r"""
    Calculate Bi-Elliptic transfer $\Delta V$ and time.

    More efficient than Hohmann if $r_2/r_1 > 15.58$.

    Parameters
    ----------
    r1 : float
        Initial radius (km).
    r2 : float
        Final radius (km).
    rb : float
        Intermediate apogee radius (km).
    mu : float, optional
        Gravitational parameter ($km^3/s^2$).

    Returns
    -------
    tuple[float, float, float, float]
        (dv1, dv2, dv3, total_time).
    """
    r1_arr = np.asarray(r1, dtype=float)
    r2_arr = np.asarray(r2, dtype=float)
    rb_arr = np.asarray(rb, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)

    # First transfer: r1 to rb
    a1 = (r1_arr + rb_arr) / 2.0
    v_c1 = np.sqrt(mu_arr / r1_arr)
    v_trans1_p = np.sqrt(mu_arr * (2 / r1_arr - 1 / a1))
    dv1 = abs(v_trans1_p - v_c1)

    v_trans1_a = np.sqrt(mu_arr * (2 / rb_arr - 1 / a1))

    # Second transfer: rb to r2
    a2 = (r2_arr + rb_arr) / 2.0
    v_trans2_a = np.sqrt(mu_arr * (2 / rb_arr - 1 / a2))
    dv2 = abs(v_trans2_a - v_trans1_a)  # Burn at apoapsis rb

    v_trans2_p = np.sqrt(mu_arr * (2 / r2_arr - 1 / a2))
    v_c2 = np.sqrt(mu_arr / r2_arr)
    dv3 = abs(v_c2 - v_trans2_p)

    # Total time
    t1 = np.pi * np.sqrt(a1**3 / mu_arr)
    t2 = np.pi * np.sqrt(a2**3 / mu_arr)
    t_total = t1 + t2

    return float(dv1), float(dv2), float(dv3), float(t_total)


def phasing_maneuver(
    a: float, t_phase: float, mu: float = 398600.4418
) -> tuple[float, float]:
    """
    Calculate Delta-V required for an orbital phasing maneuver.

    Parameters
    ----------
    a : float
        Semi-major axis of the current circular orbit (km).
    t_phase : float
        Desired time difference to generate (s).
        Positive to wait (increase period), negative to catch up.
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple[float, float]
        - total_dv: Total Delta-V (entry + exit) (km/s).
        - t_wait: Duration of the phasing orbit (s).
    """
    t_curr = 2 * np.pi * np.sqrt(a**3 / mu)
    t_phasing = t_curr + t_phase  # phasing orbit period
    a_phasing = (mu * (t_phasing / (2 * np.pi)) ** 2) ** (1 / 3)
    v_c = np.sqrt(mu / a)
    v_phasing = np.sqrt(mu * (2 / a - 1 / a_phasing))
    dv_burn = abs(v_phasing - v_c)
    total_dv = 2 * dv_burn  # two burns: entry and exit
    return total_dv, t_phasing


def plane_change(v_mag: float, delta_i: float) -> float:
    """
    Calculate Delta-V for a simple inclination plane change maneuver.

    Parameters
    ----------
    v_mag : float
        Current velocity magnitude (km/s).
    delta_i : float
        Plane change angle (radians).

    Returns
    -------
    float
        Delta-V required (km/s).
    """
    return float(2 * v_mag * np.sin(delta_i / 2.0))


def combined_plane_change(v1: float, v2: float, delta_i: float) -> float:
    """
    Calculate Delta-V for a combined maneuver (velocity magnitude + inclination).

    Parameters
    ----------
    v1 : float
        Initial velocity magnitude (km/s).
    v2 : float
        Final velocity magnitude (km/s).
    delta_i : float
        Plane change angle (radians).

    Returns
    -------
    float
        Delta-V required (km/s).
    """
    return float(np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(delta_i)))


def delta_v_budget(initial_mass: float, dv_total: float, isp: float) -> float:
    """
    Calculate required propellant mass using the Tsiolkovsky rocket equation.

    Parameters
    ----------
    initial_mass : float
        Initial mass of the spacecraft (kg).
    dv_total : float
        Total Delta-V required (km/s).
    isp : float
        Specific impulse of the propulsion system (s).

    Returns
    -------
    float
        Propellant mass required (kg).
    """
    g0 = 0.00980665  # standard gravity in km/s^2
    mass_ratio = np.exp(dv_total / (isp * g0))
    final_mass = initial_mass / mass_ratio
    return float(initial_mass - final_mass)


def raan_change(v_mag: float, inc: float, delta_raan: float) -> float:
    """
    Calculate Delta-V for a Right Ascension of Ascending Node (RAAN) change.

    Assumes circular orbit and maneuver performed at the poles for max efficiency.

    Parameters
    ----------
    v_mag : float
        Orbital velocity (km/s).
    inc : float
        Inclination (radians).
    delta_raan : float
        Desired RAAN change (radians).

    Returns
    -------
    float
        Delta-V required (km/s).
    """
    # cos(alpha) = cos^2(i) + sin^2(i) * cos(delta_raan)
    cos_alpha = np.cos(inc) ** 2 + np.sin(inc) ** 2 * np.cos(delta_raan)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    return float(2 * v_mag * np.sin(alpha / 2.0))


def optimal_combined_maneuver(
    r1: float, r2: float, delta_i: float, mu: float = 398600.4418
) -> tuple[float, float, float]:
    """
    Find optimal split of inclination change between two burns of a Hohmann transfer.

    Minimizes total Delta-V by balancing the plane change between higher and
    lower velocity points.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius (km).
    r2 : float
        Final circular orbit radius (km).
    delta_i : float
        Total inclination change (radians).
    mu : float, optional
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple[float, float, float]
        - dv_total: Minimum total Delta-V (km/s).
        - di1: Inclination change at first burn (radians).
        - di2: Inclination change at second burn (radians).
    """
    a_trans = (r1 + r2) / 2.0
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    v_trans_p = np.sqrt(mu * (2 / r1 - 1 / a_trans))
    v_trans_a = np.sqrt(mu * (2 / r2 - 1 / a_trans))

    from scipy.optimize import minimize_scalar

    def objective(di1: float) -> float:
        dv1 = np.sqrt(v_c1**2 + v_trans_p**2 - 2 * v_c1 * v_trans_p * np.cos(di1))
        dv2 = np.sqrt(v_trans_a**2 + v_c2**2 - 2 * v_trans_a * v_c2 * np.cos(delta_i - di1))
        return float(dv1 + dv2)

    res = minimize_scalar(objective, bounds=(0, delta_i), method="bounded")
    di1_opt = float(res.x)
    di2_opt = delta_i - di1_opt

    return float(res.fun), di1_opt, di2_opt




