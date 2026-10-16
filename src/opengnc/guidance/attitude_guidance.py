"""
Attitude guidance laws for Nadir, Sun, and Target pointing, and trajectory planning.
"""

import numpy as np

from opengnc.utils.quat_utils import axis_angle_to_quat, quat_normalize


def nadir_pointing_reference(pos_eci: np.ndarray, vel_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion for Nadir pointing.

    The target frame is defined as:
    - Z-axis: Points towards Earth center (Nadir).
    - Y-axis: Aligned with the negative of the orbit normal.
    - X-axis: Completes the right-handed frame (approximately velocity direction).

    Parameters
    ----------
    pos_eci : np.ndarray
        ECI position vector [m] (3,).
    vel_eci : np.ndarray
        ECI velocity vector [m/s] (3,).

    Returns
    -------
    np.ndarray
        Reference quaternion [x, y, z, w].
    """
    pos_vec = np.asarray(pos_eci)
    vel_vec = np.asarray(vel_eci)

    # Z-axis (Nadir): Towards Earth center
    z_axis = -pos_vec / np.linalg.norm(pos_vec)

    # Y-axis: Negative of Orbit normal
    orb_normal = np.cross(pos_vec, vel_vec)
    y_axis = -orb_normal / np.linalg.norm(orb_normal)

    # X-axis: Complete the frame
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # DCM from ECI to Body (R_eb)
    rmat = np.array([x_axis, y_axis, z_axis])

    return _rmat_to_quat(rmat)


def sun_pointing_reference(sun_vec_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion for Sun pointing.

    Primary objective is to align the spacecraft body X-axis with the Sun vector.
    Secondary objective is to minimize rotation about the Sun vector.

    Parameters
    ----------
    sun_vec_eci : np.ndarray
        ECI Sun direction vector (3,).

    Returns
    -------
    np.ndarray
        Reference quaternion [x, y, z, w].
    """
    sun_dir = sun_vec_eci / np.linalg.norm(sun_vec_eci)

    # Shortest rotation from ECI X to Sun vector
    eci_x = np.array([1.0, 0.0, 0.0])

    dot_prod = np.dot(eci_x, sun_dir)
    if dot_prod > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot_prod < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])

    rot_axis = np.cross(eci_x, sun_dir)
    rot_angle = np.arccos(dot_prod)

    return axis_angle_to_quat(rot_axis / np.linalg.norm(rot_axis) * rot_angle)


def target_tracking_reference(pos_eci: np.ndarray, target_pos_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion to track a target position.

    The target frame is defined with the Z-axis aligned with the boresight (towards target).

    Parameters
    ----------
    pos_eci : np.ndarray
        ECI position vector of the spacecraft [m] (3,).
    target_pos_eci : np.ndarray
        ECI position vector of the target [m] (3,).

    Returns
    -------
    np.ndarray
        Reference quaternion [x, y, z, w].
    """
    rel_pos = target_pos_eci - pos_eci
    z_axis = rel_pos / np.linalg.norm(rel_pos)

    # Auxiliary vector for frame construction
    aux_z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, aux_z)) > 0.99:
        aux_z = np.array([1.0, 0.0, 0.0])

    y_axis = np.cross(z_axis, aux_z)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rmat = np.vstack([x_axis, y_axis, z_axis])
    return _rmat_to_quat(rmat)


def eigenaxis_slew_path_planning(
    q_initial: np.ndarray, q_final: np.ndarray, time_span: np.ndarray
) -> list[np.ndarray]:
    """
    Generate an eigenaxis slew profile using Spherical Linear Interpolation (SLERP).

    Parameters
    ----------
    q_initial : np.ndarray
        Initial quaternion [x, y, z, w].
    q_final : np.ndarray
        Final quaternion [x, y, z, w].
    time_span : np.ndarray
        Normalized time values [0, 1] (N,).

    Returns
    -------
    list[np.ndarray]
        List of interpolated quaternions along the path.
    """
    return [attitude_blending(q_initial, q_final, float(t)) for t in time_span]


def attitude_blending(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """
    SLERP (Spherical Linear Interpolation) between two orientation quaternions.

    Parameters
    ----------
    q1 : np.ndarray
        First quaternion [x, y, z, w].
    q2 : np.ndarray
        Second quaternion [x, y, z, w].
    alpha : float
        Blending factor [0, 1].

    Returns
    -------
    np.ndarray
        Blended/interpolated quaternion [x, y, z, w].
    """
    quat1 = quat_normalize(q1)
    quat2 = quat_normalize(q2)

    dot_prod = np.dot(quat1, quat2)

    # Ensure shortest path
    if dot_prod < 0.0:
        quat1 = -quat1
        dot_prod = -dot_prod

    if dot_prod > 0.9995:
        # Linear interpolation for very small angles
        res = quat1 + alpha * (quat2 - quat1)
        return quat_normalize(res)

    theta_0 = np.arccos(dot_prod)
    theta = theta_0 * alpha

    sin_theta_0 = np.sin(theta_0)
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot_prod * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return np.asarray((s0 * quat1) + (s1 * quat2))


def _rmat_to_quat(r_mat: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion using Shepperd's algorithm.

    Parameters
    ----------
    r_mat : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    np.ndarray
        Quaternion [x, y, z, w].
    """
    tr_val = np.trace(r_mat)
    if tr_val > 0:
        scale_s = np.sqrt(tr_val + 1.0) * 2
        quat_w = 0.25 * scale_s
        quat_x = (r_mat[2, 1] - r_mat[1, 2]) / scale_s
        quat_y = (r_mat[0, 2] - r_mat[2, 0]) / scale_s
        quat_z = (r_mat[1, 0] - r_mat[0, 1]) / scale_s
    elif (r_mat[0, 0] > r_mat[1, 1]) and (r_mat[0, 0] > r_mat[2, 2]):
        scale_s = np.sqrt(1.0 + r_mat[0, 0] - r_mat[1, 1] - r_mat[2, 2]) * 2
        quat_w = (r_mat[2, 1] - r_mat[1, 2]) / scale_s
        quat_x = 0.25 * scale_s
        quat_y = (r_mat[0, 1] + r_mat[1, 0]) / scale_s
        quat_z = (r_mat[0, 2] + r_mat[2, 0]) / scale_s
    elif r_mat[1, 1] > r_mat[2, 2]:
        scale_s = np.sqrt(1.0 + r_mat[1, 1] - r_mat[0, 0] - r_mat[2, 2]) * 2
        quat_w = (r_mat[0, 2] - r_mat[2, 0]) / scale_s
        quat_x = (r_mat[0, 1] + r_mat[1, 0]) / scale_s
        quat_y = 0.25 * scale_s
        quat_z = (r_mat[1, 2] + r_mat[2, 1]) / scale_s
    else:
        scale_s = np.sqrt(1.0 + r_mat[2, 2] - r_mat[0, 0] - r_mat[1, 1]) * 2
        quat_w = (r_mat[1, 0] - r_mat[0, 1]) / scale_s
        quat_x = (r_mat[0, 2] + r_mat[2, 0]) / scale_s
        quat_y = (r_mat[1, 2] + r_mat[2, 1]) / scale_s
        quat_z = 0.25 * scale_s

    return np.array([quat_x, quat_y, quat_z, quat_w])
