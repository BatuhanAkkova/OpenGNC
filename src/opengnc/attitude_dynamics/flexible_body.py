"""
Flexible body dynamics propagation and coupling models.
"""

import numpy as np


def flexible_body_dynamics(
    eta: np.ndarray,
    eta_dot: np.ndarray,
    omega_dot: np.ndarray,
    natural_freqs: np.ndarray,
    damping_ratios: np.ndarray,
    modal_influence: np.ndarray,
) -> np.ndarray:
    r"""
    Compute modal acceleration $\ddot{\eta}$.

    Modal Equation:
    $\ddot{\eta}_i + 2\zeta_i\omega_{n,i}\dot{\eta}_i + \omega_{n,i}^2\eta_i = \mathbf{\Phi}_i \cdot \dot{\omega}$

    Parameters
    ----------
    eta : np.ndarray
        Modal displacements (n_modes,).
    eta_dot : np.ndarray
        Modal velocities (n_modes,).
    omega_dot : np.ndarray
        Rigid body angular acceleration (3,) (rad/s$^2$).
    natural_freqs : np.ndarray
        Natural frequencies $\omega_{n,i}$ (n_modes,) (rad/s).
    damping_ratios : np.ndarray
        Damping ratios $\zeta_i$ (n_modes,).
    modal_influence : np.ndarray
        Influence matrix $\mathbf{\Phi}$ (n_modes, 3).

    Returns
    -------
    np.ndarray
        Modal acceleration $\ddot{\eta}$ (n_modes,) (rad/s$^2$).
    """
    eta_val = np.asarray(eta)
    edot_val = np.asarray(eta_dot)
    odot_val = np.asarray(omega_dot)

    # Force/Torque coupling term: F_modal = Phi * omega_dot
    coupling = modal_influence @ odot_val

    # Damping: 2 * zeta * omega_n * eta_dot
    damping = 2 * damping_ratios * natural_freqs * edot_val

    # Stiffness: omega_n^2 * eta
    stiffness = (natural_freqs**2) * eta_val

    return np.asarray(coupling - damping - stiffness)


def coupled_flexible_rigid_dynamics(
    J_rigid: np.ndarray,
    omega: np.ndarray,
    torque: np.ndarray,
    eta: np.ndarray,
    eta_dot: np.ndarray,
    natural_freqs: np.ndarray,
    damping_ratios: np.ndarray,
    modal_influence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute coupled rigid-flexible body dynamics.

    Augmented System Projection:
    $\begin{bmatrix} \mathbf{J} & \mathbf{\Phi}^T \\ \mathbf{\Phi} & \mathbf{I} \end{bmatrix} \begin{bmatrix} \dot{\omega} \\ \ddot{\eta} \end{bmatrix} = \begin{bmatrix} \tau - \omega \times (\mathbf{J} \omega) \\ -2\zeta\omega_n\dot{\eta} - \omega_n^2\eta \end{bmatrix}$

    Parameters
    ----------
    J_rigid : np.ndarray
        Rigid inertia tensor (3, 3) ($kg \cdot m^2$).
    omega : np.ndarray
        Angular velocity (3,) (rad/s).
    torque : np.ndarray
        External torque (3,) (Nm).
    eta : np.ndarray
        Modal displacements (n_modes,).
    eta_dot : np.ndarray
        Modal velocities (n_modes,).
    natural_freqs : np.ndarray
        Natural frequencies (n_modes,) (rad/s).
    damping_ratios : np.ndarray
        Damping ratios (n_modes,).
    modal_influence : np.ndarray
        Influence matrix (n_modes, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (omega_dot, eta_ddot).
    """
    jr = np.asarray(J_rigid)
    w = np.asarray(omega)
    tq = np.asarray(torque)
    n_modes = len(natural_freqs)

    # 1. Build augmented mass matrix M_aug
    m_aug = np.zeros((3 + n_modes, 3 + n_modes))
    m_aug[:3, :3] = jr
    m_aug[:3, 3:] = modal_influence.T
    m_aug[3:, :3] = modal_influence
    m_aug[3:, 3:] = np.eye(n_modes)

    # 2. Build Right-Hand Side
    rhs = np.zeros(3 + n_modes)
    rhs[:3] = tq - np.cross(w, jr @ w)
    rhs[3:] = -(2 * damping_ratios * natural_freqs * eta_dot + (natural_freqs**2) * eta)

    # 3. Solve system
    sol = np.linalg.solve(m_aug, rhs)

    return np.asarray(sol[:3]), np.asarray(sol[3:])
