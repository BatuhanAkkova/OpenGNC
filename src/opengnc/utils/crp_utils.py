"""
Classical Rodrigues Parameters (CRP) kinematics and composition.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def quat_to_crp(q: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to Classical Rodrigues Parameters (CRP/Gibbs).

    Mathematical form: $\mathbf{q}_{vec} / q_w$.
    Singular for 180-degree rotations ($q_w = 0$).

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        CRP vector $[q_1, q_2, q_3]^T$.

    Raises
    ------
    ValueError
        If the rotation is 180 degrees.
    """
    qv = np.asarray(q)
    x, y, z, w = qv
    if abs(w) < 1e-12:
        raise ValueError("CRP is singular for 180-degree rotations.")
    return cast(np.ndarray, np.array([x, y, z]) / w)


def crp_to_quat(q_crp: np.ndarray) -> np.ndarray:
    """
    Convert Classical Rodrigues Parameters to a quaternion.

    Parameters
    ----------
    q_crp : np.ndarray
        CRP vector $[q_1, q_2, q_3]^T$.

    Returns
    -------
    np.ndarray
        Unit quaternion [x, y, z, w].
    """
    cv = np.asarray(q_crp)
    norm_sq = np.sum(cv**2)
    den = np.sqrt(1.0 + norm_sq)
    return cast(np.ndarray, np.array([cv[0] / den, cv[1] / den, cv[2] / den, 1.0 / den]))


def crp_to_dcm(q_crp: np.ndarray) -> np.ndarray:
    """
    Convert Classical Rodrigues Parameters to a Direction Cosine Matrix.

    Parameters
    ----------
    q_crp : np.ndarray
        CRP vector $[q_1, q_2, q_3]^T$.

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix.
    """
    cv = np.asarray(q_crp)
    norm_sq = np.sum(cv**2)
    q1, q2, q3 = cv

    s_mat = np.array([
        [0.0, -q3,  q2],
        [q3,  0.0, -q1],
        [-q2, q1,  0.0]
    ])

    return cast(np.ndarray, np.eye(3) + (2.0 * s_mat @ s_mat + 2.0 * s_mat) / (1.0 + norm_sq))


def crp_addition(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r"""
    Compose two rotations represented by Classical Rodrigues Parameters.

    $\mathbf{q}_{res} = \frac{\mathbf{q}_1 + \mathbf{q}_2 + \mathbf{q}_2 \times \mathbf{q}_1}{1 - \mathbf{q}_1 \cdot \mathbf{q}_2}$

    Parameters
    ----------
    q1 : np.ndarray
        Initial CRP vector.
    q2 : np.ndarray
        Secondary CRP vector.

    Returns
    -------
    np.ndarray
        Resultant CRP vector.

    Raises
    ------
    ValueError
        If the denominator is near zero.
    """
    qv1 = np.asarray(q1)
    qv2 = np.asarray(q2)

    dot = np.dot(qv1, qv2)
    if abs(1.0 - dot) < 1e-12:
        raise ValueError("CRP addition is singular (rotation equals 180 degrees).")

    return cast(np.ndarray, (qv1 + qv2 + np.cross(qv2, qv1)) / (1.0 - dot))




