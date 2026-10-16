"""
Fuel slosh dynamics using equivalent pendulum models.
"""

import numpy as np


def fuel_slosh_dynamics(
    theta: float,
    theta_dot: float,
    omega: np.ndarray,
    omega_dot: np.ndarray,
    L: float,
    r_base: np.ndarray,
    g_equiv: np.ndarray,
) -> float:
    r"""
    Compute slosh pendulum angular acceleration $\ddot{\theta}$.

    The pendulum is subject to base acceleration:
    $\mathbf{a}_{base} = \mathbf{g}_{eq} - \dot{\omega} \times \mathbf{r}_b - \omega \times (\omega \times \mathbf{r}_b)$

    Parameters
    ----------
    theta : float
        Pendulum angle (rad).
    theta_dot : float
        Pendulum rate (rad/s).
    omega : np.ndarray
        Spacecraft angular velocity (3,) (rad/s).
    omega_dot : np.ndarray
        Spacecraft angular acceleration (3,) (rad/s$^2$).
    L : float
        Pendulum length (m).
    r_base : np.ndarray
        Pivot location relative to spacecraft CM (3,) (m).
    g_equiv : np.ndarray
        Effective gravity/acceleration vector (3,) (m/s$^2$).

    Returns
    -------
    float
        Pendulum acceleration $\ddot{\theta}$ (rad/s$^2$).
    """
    w = np.asarray(omega)
    odot = np.asarray(omega_dot)
    rb = np.asarray(r_base)
    ge = np.asarray(g_equiv)

    # 1. Acceleration at the pivot (base) in body frame
    # a_base = g - omega_dot x r - omega x (omega x r)
    a_base = ge - np.cross(odot, rb) - np.cross(w, np.cross(w, rb))

    # 2. Transverse unit vector: e_theta = [cos(theta), 0, sin(theta)]
    e_theta = np.array([np.cos(theta), 0, np.sin(theta)])

    # 3. Torque balance / dynamics: L * theta_ddot = a_base \cdot e_theta
    return float(np.dot(a_base, e_theta) / L)


def fuel_slosh_torque(
    m_p: float, L: float, theta: float, theta_dot: float, theta_ddot: float, r_base: np.ndarray
) -> np.ndarray:
    """
    Compute reaction torque from slosh bob.

    Parameters
    ----------
    m_p : float
        Slosh mass (kg).
    L : float
        Pendulum length (m).
    theta : float
        Pendulum angle (rad).
    theta_dot : float
        Pendulum rate (rad/s).
    theta_ddot : float
        Pendulum acceleration (rad/s$^2$).
    r_base : np.ndarray
        Pivot location (3,) (m).

    Returns
    -------
    np.ndarray
        Reaction torque vector (3,) (Nm).
    """
    rb = np.asarray(r_base)

    # Relative unit vectors for pendulum bob
    e_r = np.array([np.sin(theta), 0, -np.cos(theta)])
    e_theta = np.array([np.cos(theta), 0, np.sin(theta)])

    # Bob relative acceleration: a_rel = L*theta_ddot*e_theta - L*theta_dot^2*e_r
    a_rel = L * theta_ddot * e_theta - L * (theta_dot**2) * e_r

    # Reaction force: F_react = -m_p * a_rel
    force_react = -m_p * a_rel

    # Torque: r_bob x F_react
    r_bob = rb + L * e_r
    return np.cross(r_bob, force_react)
