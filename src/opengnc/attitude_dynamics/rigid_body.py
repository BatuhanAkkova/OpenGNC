"""
Rigid body Attitude dynamics based on Euler's equations of motion.
"""

from __future__ import annotations

import numpy as np


def euler_equations(J: np.ndarray, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
    r"""
    Compute rigid body angular acceleration via Euler's equations.

    Equation of Motion:
    $\mathbf{J} \dot{\omega} + \mathbf{\omega} \times (\mathbf{J} \omega) = \mathbf{\tau}$

    Parameters
    ----------
    J : np.ndarray
        Inertia tensor ($3 \times 3$) ($kg \cdot m^2$).
    omega : np.ndarray
        Angular velocity vector (3,) (rad/s).
    torque : np.ndarray
        Net external torque vector (3,) (Nm).

    Returns
    -------
    np.ndarray
        Angular acceleration $\dot{\omega}$ (3,) (rad/s$^2$).

    Raises
    ------
    ValueError
        If input dimensions are invalid.
    """
    # Input validation
    if J.shape != (3, 3):
        raise ValueError(f"Inertia tensor J must be shape (3, 3), got {J.shape}")
    if omega.shape != (3,):
        raise ValueError(f"Angular velocity omega must be shape (3,), got {omega.shape}")
    if torque.shape != (3,):
        raise ValueError(f"Torque vector must be shape (3,), got {torque.shape}")

    # Solve for omega_dot: J * omega_dot = torque - omega x (J * omega)
    h_vec = J @ omega
    gyro_term = np.cross(omega, h_vec)
    rhs = torque - gyro_term

    return np.linalg.solve(J, rhs)
