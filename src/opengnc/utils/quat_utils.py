"""
Quaternion kinematics and math utilities.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    r"""Normalize a quaternion to unit length."""
    qv = np.asarray(q)
    norm = np.linalg.norm(qv)
    if norm < 1e-15:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return cast(np.ndarray, qv / norm)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate of a quaternion."""
    qv = np.asarray(q)
    return cast(np.ndarray, np.array([-qv[0], -qv[1], -qv[2], qv[3]]))


def quat_norm(q: np.ndarray) -> float:
    """Compute the norm of a quaternion."""
    return float(np.linalg.norm(np.asarray(q)))


def quat_mult(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    r"""Multiply two quaternions using the Hamilton product."""
    ql = np.asarray(q_left)
    qr = np.asarray(q_right)
    x1, y1, z1, w1 = ql
    x2, y2, z2, w2 = qr
    return cast(
        np.ndarray,
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        ),
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Compute the inverse of a quaternion."""
    qv = np.asarray(q)
    norm = quat_norm(qv)
    if norm < 1e-15:
        raise ValueError("Cannot invert a zero-length quaternion.")
    return cast(np.ndarray, quat_conj(qv) / norm**2)


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Rotate a 3D vector by a quaternion."""
    qv = np.asarray(q)
    vv = np.asarray(v)
    v_ext = np.array([vv[0], vv[1], vv[2], 0.0])
    q_inv = quat_conj(qv)
    res = quat_mult(quat_mult(qv, v_ext), q_inv)
    return cast(np.ndarray, res[0:3])


def quat_to_rmat(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a 3x3 direction cosine matrix."""
    qv = np.asarray(q)
    x_val, y_val, z_val, w_val = qv
    return cast(
        np.ndarray,
        np.array(
            [
                [
                    1 - 2 * y_val**2 - 2 * z_val**2,
                    2 * x_val * y_val - 2 * z_val * w_val,
                    2 * x_val * z_val + 2 * y_val * w_val,
                ],
                [
                    2 * x_val * y_val + 2 * z_val * w_val,
                    1 - 2 * x_val**2 - 2 * z_val**2,
                    2 * y_val * z_val - 2 * x_val * w_val,
                ],
                [
                    2 * x_val * z_val - 2 * y_val * w_val,
                    2 * y_val * z_val + 2 * x_val * w_val,
                    1 - 2 * x_val**2 - 2 * y_val**2,
                ],
            ]
        ),
    )


def axis_angle_to_quat(axis: np.ndarray, angle: float | None = None) -> np.ndarray:
    r"""Convert an axis-angle pair or rotation vector to a quaternion."""
    av = np.asarray(axis)
    if angle is None:
        norm = float(np.linalg.norm(av))
        if norm < 1e-15:
            return cast(np.ndarray, np.array([0.0, 0.0, 0.0, 1.0]))
        u_vec = av / norm
        theta = norm
    else:
        norm = float(np.linalg.norm(av))
        if norm < 1e-15:
            return cast(np.ndarray, np.array([0.0, 0.0, 0.0, 1.0]))
        u_vec = av / norm
        theta = float(angle)

    s_val = np.sin(theta / 2.0)
    c_val = np.cos(theta / 2.0)
    return cast(np.ndarray, np.array([u_vec[0] * s_val, u_vec[1] * s_val, u_vec[2] * s_val, c_val]))


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    r"""
    Create a 3x3 skew-symmetric matrix from a vector.

    This is the matrix ``S(v)`` such that ``S(v) @ w`` equals ``v x w``.
    """
    vv = np.asarray(v)
    return cast(
        np.ndarray,
        np.array(
            [
                [0.0, -vv[2], vv[1]],
                [vv[2], 0.0, -vv[0]],
                [-vv[1], vv[0], 0.0],
            ]
        ),
    )
