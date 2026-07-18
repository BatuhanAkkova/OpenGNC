"""
Modified Rodrigues Parameters (MRP) kinematics and math utilities.
"""

from typing import cast

import numpy as np


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to Modified Rodrigues Parameters (MRP).

    Mathematical form: $\sigma = \mathbf{q}_{vec} / (1 + q_w)$.
    Singular for 360-degree rotations ($q_w = -1$).

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        MRP vector $[\sigma_1, \sigma_2, \sigma_3]^T$.
    """
    qv = np.asarray(q)
    x, y, z, w = qv
    # Handle singularity by adding a small epsilon or implying shadow set
    den = 1.0 + w
    if abs(den) < 1e-12:
        # Fallback to a very large but finite MRP or notify
        # In practice, we use shadow sets before this happens.
        return cast(np.ndarray, np.array([x, y, z]) * 1e12)
    return cast(np.ndarray, np.array([x, y, z]) / den)


def mrp_to_quat(sigma: np.ndarray) -> np.ndarray:
    r"""
    Convert Modified Rodrigues Parameters to a quaternion.

    Parameters
    ----------
    sigma : np.ndarray
        MRP vector $[\sigma_1, \sigma_2, \sigma_3]^T$.

    Returns
    -------
    np.ndarray
        Unit quaternion [x, y, z, w].
    """
    sv = np.asarray(sigma)
    sigma_sq = np.sum(sv**2)
    w = (1.0 - sigma_sq) / (1.0 + sigma_sq)
    xyz = 2.0 * sv / (1.0 + sigma_sq)
    return cast(np.ndarray, np.array([xyz[0], xyz[1], xyz[2], w]))


def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    r"""
    Convert Modified Rodrigues Parameters to a Direction Cosine Matrix.

    Parameters
    ----------
    sigma : np.ndarray
        MRP vector $[\sigma_1, \sigma_2, \sigma_3]^T$.

    Returns
    -------
    np.ndarray
        3x3 Direction Cosine Matrix.
    """
    sv = np.asarray(sigma)
    sigma_sq = np.sum(sv**2)
    s1, s2, s3 = sv

    s_mat = np.array([
        [0.0, -s3,  s2],
        [s3,  0.0, -s1],
        [-s2, s1,  0.0]
    ])

    den = (1.0 + sigma_sq)**2
    return cast(np.ndarray, np.eye(3) + (8.0 * s_mat @ s_mat + 4.0 * (1.0 - sigma_sq) * s_mat) / den)


def get_shadow_mrp(sigma: np.ndarray) -> np.ndarray:
    r"""
    Return the shadow set for the given MRP.

    The shadow set represents the same rotation but stays within the
    unit sphere ($|\sigma| \le 1$).
    $\sigma_{shadow} = -\sigma / \|\sigma\|^2$.

    Parameters
    ----------
    sigma : np.ndarray
        MRP vector $[\sigma_1, \sigma_2, \sigma_3]^T$.

    Returns
    -------
    np.ndarray
        Shadow MRP vector.
    """
    sv = np.asarray(sigma)
    sigma_sq = np.sum(sv**2)
    if sigma_sq < 1e-15:
        return cast(np.ndarray, np.zeros(3))
    return cast(np.ndarray, -sv / sigma_sq)


def check_mrp_switching(sigma: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Ensure the MRP magnitude remains below a threshold by switching to the shadow set.

    Typically used to maintain the MRP within the unit sphere ($threshold=1.0$).

    Parameters
    ----------
    sigma : np.ndarray
        Current MRP vector.
    threshold : float, optional
        Magnitude threshold for switching. Default is 1.0.

    Returns
    -------
    np.ndarray
        Standard or shadow MRP vector.
    """
    sv = np.asarray(sigma)
    if np.linalg.norm(sv) > threshold:
        return get_shadow_mrp(sv)
    return cast(np.ndarray, sv)




