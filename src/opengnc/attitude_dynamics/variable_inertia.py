"""
Attitude dynamics with time-varying inertia tensors.
"""

import numpy as np


def variable_inertia_euler_equations(
    J: np.ndarray, J_dot: np.ndarray, omega: np.ndarray, torque: np.ndarray
) -> np.ndarray:
    r"""
    Compute $\dot{\omega}$ for a body with time-varying inertia.

    Equation of Motion:
    $\mathbf{J} \dot{\omega} + \dot{\mathbf{J}} \omega + \omega \times (\mathbf{J} \omega) = \mathbf{\tau}$

    Parameters
    ----------
    J : np.ndarray
        Current inertia tensor ($3 \times 3$) ($kg \cdot m^2$).
    J_dot : np.ndarray
        Inertia derivative ($3 \times 3$) ($kg \cdot m^2 / s$).
    omega : np.ndarray
        Angular velocity (3,) (rad/s).
    torque : np.ndarray
        External torque (3,) (Nm).

    Returns
    -------
    np.ndarray
        Angular acceleration $\dot{\omega}$ (3,) (rad/s$^2$).
    """
    j_mat = np.asarray(J)
    jd_mat = np.asarray(J_dot)
    w = np.asarray(omega)
    tq = np.asarray(torque)

    # 1. Right-hand side: RHS = tau - J_dot*omega - omega x (J*omega)
    rhs = tq - jd_mat @ w - np.cross(w, j_mat @ w)

    # 2. Solve for omega_dot: J * omega_dot = RHS
    return np.linalg.solve(j_mat, rhs)


def mass_depletion_J_dot(
    J_nominal: np.ndarray, m_initial: float, dm_dt: float, r_point: np.ndarray
) -> np.ndarray:
    r"""
    Model $\dot{\mathbf{J}}$ due to point-mass depletion.

    $\dot{\mathbf{J}} = \dot{m} [ (\mathbf{r}^T \mathbf{r}) \mathbf{I} - \mathbf{r} \mathbf{r}^T ]$

    Parameters
    ----------
    J_nominal : np.ndarray
        Nominal system inertia (3, 3).
    m_initial : float
        Initial segment mass (kg).
    dm_dt : float
        Mass flow rate (kg/s). Negative for depletion.
    r_point : np.ndarray
        Position of mass segment relative to system CM (3,) (m).

    Returns
    -------
    np.ndarray
        Inertia derivative $\dot{\mathbf{J}}$ (3, 3) ($kg \cdot m^2 / s$).
    """
    r_vec = np.asarray(r_point)
    r_sq = float(np.dot(r_vec, r_vec))
    r_outer = np.outer(r_vec, r_vec)

    # dot(J) = dot(m) * [ (r^2 * I) - (r outer r) ]
    return dm_dt * (r_sq * np.eye(3) - r_outer)
