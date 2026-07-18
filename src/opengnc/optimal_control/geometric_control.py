"""
Geometric Controller on SO(3) for attitude tracking.
"""


from typing import cast

import numpy as np


def vee_map(R: np.ndarray) -> np.ndarray:
    r"""
    Map a $3\times 3$ skew-symmetric matrix to its corresponding $3\times 1$ vector.

    Parameters
    ----------
    R : np.ndarray
        $3\times 3$ skew-symmetric matrix.

    Returns
    -------
    np.ndarray
        $3\times 1$ vector.
    """
    return cast(np.ndarray, np.array([R[2, 1], R[0, 2], R[1, 0]]))


def hat_map(v: np.ndarray) -> np.ndarray:
    r"""
    Map a $3\times 1$ vector to its corresponding $3\times 3$ skew-symmetric matrix.

    Parameters
    ----------
    v : np.ndarray
        $3\times 1$ vector.

    Returns
    -------
    np.ndarray
        $3\times 3$ skew-symmetric matrix.
    """
    v_vec = np.asarray(v).flatten()
    return cast(np.ndarray, np.array([
        [0.0, -v_vec[2], v_vec[1]],
        [v_vec[2], 0.0, -v_vec[0]],
        [-v_vec[1], v_vec[0], 0.0]
    ]))


class GeometricController:
    """
    Geometric Controller on the Special Orthogonal Group SO(3).

    Implements a tracking controller directly on the rotation matrix manifold,
    avoiding singularities and unwinding issues associated with Euler angles
    and quaternions.

    Ref: Lee, T., Leok, M., & McClamroch, N. H. (2010). Geometric Tracking
    Control of a Quadrotor UAV on SE(3).

    Parameters
    ----------
    J : np.ndarray
        Spacecraft inertia tensor matrix (3x3) [kg*m^2].
    kR : float
        Attitude error gain (proportional).
    kW : float
        Angular velocity error gain (derivative).
    """

    def __init__(self, J: np.ndarray, kR: float, kW: float) -> None:
        """Initialize geometric controller gains and inertia."""
        self.J = np.asarray(J)
        self.kR = float(kR)
        self.kW = float(kW)

    def compute_control(
        self,
        R: np.ndarray,
        omega: np.ndarray,
        R_d: np.ndarray,
        omega_d: np.ndarray,
        d_omega_d: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the optimal control torque on SO(3).

        Parameters
        ----------
        R : np.ndarray
            Current rotation matrix (3x3).
        omega : np.ndarray
            Current angular velocity in body frame (3,).
        R_d : np.ndarray
            Desired rotation matrix (3x3).
        omega_d : np.ndarray
            Desired angular velocity in body frame (3,).
        d_omega_d : np.ndarray, optional
            Desired angular acceleration (3,). Defaults to zero.

        Returns
        -------
        np.ndarray
            Control torque vector M (3,) [N*m].
        """
        R_mat = np.asarray(R)
        omega_vec = np.asarray(omega)
        Rd_mat = np.asarray(R_d)
        wd_vec = np.asarray(omega_d)
        dwd_vec = np.asarray(d_omega_d) if d_omega_d is not None else np.zeros(3)

        # 1. Attitude error: eR = 0.5 * vee(Rd^T * R - R^T * Rd)
        error_mat = Rd_mat.T @ R_mat - R_mat.T @ Rd_mat
        eR = 0.5 * vee_map(error_mat)

        # 2. Angular velocity error: eW = omega - R^T * Rd * omega_d
        R_rel = R_mat.T @ Rd_mat
        eW = omega_vec - R_rel @ wd_vec

        # 3. Feedforward and Gyroscopic compensation
        # term = J * (hat(omega) * R^T * Rd * omega_d - R^T * Rd * d_omega_d)
        w_skew = hat_map(omega_vec)
        acc_term = w_skew @ R_rel @ wd_vec - R_rel @ dwd_vec

        M_ff = self.J @ acc_term
        M_gyro = np.cross(omega_vec, self.J @ omega_vec)

        # Control Law: M = -kR*eR - kW*eW + gyro - ff
        return cast(np.ndarray, -self.kR * eR - self.kW * eW + M_gyro - M_ff)




