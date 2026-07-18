"""
State and attitude representation conversion utilities.
"""

from __future__ import annotations

import numpy as np

from opengnc.utils import euler_utils as eu
from opengnc.utils import quat_utils as qu


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a Direction Cosine Matrix (DCM).

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix (Body-to-ECI rotation).
    """
    return qu.quat_to_rmat(np.asarray(q))


def quat_to_euler(q: np.ndarray, sequence: str) -> np.ndarray:
    r"""
    Convert a quaternion to Euler angles in a specified sequence.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].
    sequence : str
        Rotation sequence (e.g., '321', '313').

    Returns
    -------
    np.ndarray
        Euler angles $[\theta_1, \theta_2, \theta_3]$ in radians.
    """
    dcm = quat_to_dcm(np.asarray(q))
    return eu.dcm_to_euler(dcm, sequence)


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """
    Convert a Direction Cosine Matrix to a quaternion.

    Uses Shepperd's algorithm for numerical stability.

    Parameters
    ----------
    dcm : np.ndarray
        3x3 Direction Cosine Matrix.

    Returns
    -------
    np.ndarray
        Unit quaternion [x, y, z, w].
    """
    mat = np.asarray(dcm)
    tr = np.trace(mat)

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])


def dcm_to_euler(dcm: np.ndarray, sequence: str) -> np.ndarray:
    """
    Convert a Direction Cosine Matrix to Euler angles.

    Parameters
    ----------
    dcm : np.ndarray
        3x3 Direction Cosine Matrix.
    sequence : str
        Rotation sequence (e.g., '321').

    Returns
    -------
    np.ndarray
        Euler angles in radians.
    """
    return eu.dcm_to_euler(np.asarray(dcm), sequence)


def euler_to_quat(angles: np.ndarray, sequence: str) -> np.ndarray:
    """
    Convert Euler angles to a quaternion.

    Parameters
    ----------
    angles : np.ndarray
        Euler angles in radians.
    sequence : str
        Rotation sequence.

    Returns
    -------
    np.ndarray
        Quaternion [x, y, z, w].
    """
    dcm = euler_to_dcm(np.asarray(angles), sequence)
    return dcm_to_quat(dcm)


def euler_to_dcm(angles: np.ndarray, sequence: str) -> np.ndarray:
    """
    Convert Euler angles to a Direction Cosine Matrix.

    Parameters
    ----------
    angles : np.ndarray
        Euler angles in radians.
    sequence : str
        Rotation sequence.

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix.
    """
    return eu.euler_to_dcm(np.asarray(angles), sequence)

def rot_x(theta: float) -> np.ndarray:
    """Return the x-axis rotation matrix for the given angle."""
    return eu.rot_x(theta)


def rot_y(theta: float) -> np.ndarray:
    """Return the y-axis rotation matrix for the given angle."""
    return eu.rot_y(theta)


def rot_z(theta: float) -> np.ndarray:
    """Return the z-axis rotation matrix for the given angle."""
    return eu.rot_z(theta)

