"""
Euler angle kinematics and Direction Cosine Matrix (DCM) utilities.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the X-axis.

    Parameters
    ----------
    angle : float
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return cast(np.ndarray, np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ]))


def rot_y(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the Y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return cast(np.ndarray, np.array([
        [  c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,  c]
    ]))


def rot_z(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the Z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return cast(np.ndarray, np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ]))


def euler_to_dcm(angles: np.ndarray, sequence: str) -> np.ndarray:
    r"""
    Convert Euler angles to a 3x3 Direction Cosine Matrix (DCM).

    Standard convention for satellite dynamics is the Body-to-Inertial
    rotation $R_{I/B} = R_3(\theta_3) R_2(\theta_2) R_1(\theta_1)$.

    Parameters
    ----------
    angles : np.ndarray
        Three Euler angles in radians $[\theta_1, \theta_2, \theta_3]$.
    sequence : str
        Rotation sequence (e.g., '321', '313'). Must be 3 characters.

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix.

    Raises
    ------
    ValueError
        If the sequence is invalid or length is not 3.
    """
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")

    av = np.asarray(angles)
    rot_fns = {"1": rot_x, "2": rot_y, "3": rot_z}

    try:
        r1 = rot_fns[sequence[0]](av[0])
        r2 = rot_fns[sequence[1]](av[1])
        r3 = rot_fns[sequence[2]](av[2])
    except KeyError as e:
        raise ValueError(f"Invalid axis '{e.args[0]}' in sequence '{sequence}'.") from e

    # Composite rotation: last rotation applied first in matrix product
    return cast(np.ndarray, r3 @ r2 @ r1)


def dcm_to_euler(dcm: np.ndarray, sequence: str) -> np.ndarray:
    r"""
    Extract Euler angles from a 3x3 Direction Cosine Matrix.

    Supports all 12 standard rotation sequences (Symmetric and Asymmetric).
    Handles gimbal lock singularities by setting the first angle to zero.

    Parameters
    ----------
    dcm : np.ndarray
        3x3 Direction Cosine Matrix.
    sequence : str
        Sequence of rotations (e.g., '321', '313').

    Returns
    -------
    np.ndarray
        Euler angles $[\theta_1, \theta_2, \theta_3]$ in radians.
    """
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")

    mat = np.asarray(dcm)

    # 1. Symmetric Sequences (e.g., 3-1-3, 1-2-1)
    if sequence[0] == sequence[2]:
        i = int(sequence[0]) - 1
        j = int(sequence[1]) - 1
        k = 3 - i - j

        parity = 1 if (j - i) % 3 == 1 else -1

        # Mid-angle theta2 (0 to PI)
        theta2 = np.arccos(np.clip(mat[i, i], -1.0, 1.0))

        if abs(np.sin(theta2)) < 1e-12:
            # Singularity (Gimbal Lock)
            theta1 = 0.0
            theta3 = np.arctan2(parity * mat[j, k], mat[j, j])
        else:
            theta1 = np.arctan2(mat[i, j], parity * mat[i, k])
            theta3 = np.arctan2(mat[j, i], -parity * mat[k, i])

        return cast(np.ndarray, np.array([theta1, theta2, theta3]))

    # 2. Asymmetric Sequences (e.g., 3-2-1, 1-2-3)
    else:
        i = int(sequence[0]) - 1
        j = int(sequence[1]) - 1
        k = int(sequence[2]) - 1

        parity = 1 if (j - i) % 3 == 1 else -1

        # Mid-angle theta2 (-PI/2 to PI/2)
        theta2 = np.arcsin(np.clip(-parity * mat[k, i], -1.0, 1.0))

        if abs(np.cos(theta2)) < 1e-12:
            # Singularity (Gimbal Lock)
            theta1 = 0.0
            theta3 = np.arctan2(parity * mat[j, i], mat[j, j])
        else:
            theta1 = np.arctan2(parity * mat[k, j], mat[k, k])
            theta3 = np.arctan2(parity * mat[j, i], mat[i, i])

        return cast(np.ndarray, np.array([theta1, theta2, theta3]))




