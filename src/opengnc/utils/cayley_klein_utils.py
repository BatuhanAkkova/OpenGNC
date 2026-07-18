"""
Cayley-Klein parameters for attitude representation and composition.
"""

from typing import cast

import numpy as np


def quat_to_cayley_klein(q: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to a Cayley-Klein matrix.

    Parameters
    ----------
    q : np.ndarray
        Quaternion ``[x, y, z, w]``.

    Returns
    -------
    np.ndarray
        ``2 x 2`` complex unitary matrix built from the Cayley-Klein parameters.
    """
    qv = np.asarray(q, dtype=float)
    x_val, y_val, z_val, w_val = qv
    alpha = complex(w_val, z_val)
    beta = complex(y_val, x_val)

    return cast(
        np.ndarray,
        np.array([[alpha, beta], [-np.conj(beta), np.conj(alpha)]], dtype=complex),
    )


def cayley_klein_to_quat(u_mat: np.ndarray) -> np.ndarray:
    """Convert a ``2 x 2`` Cayley-Klein matrix to a quaternion."""
    umat = np.asarray(u_mat)
    alpha = umat[0, 0]
    beta = umat[0, 1]
    return cast(np.ndarray, np.array([beta.imag, beta.real, alpha.imag, alpha.real], dtype=float))


def cayley_klein_mult(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Multiply two Cayley-Klein matrices to compose rotations."""
    return cast(np.ndarray, np.asarray(u1) @ np.asarray(u2))
