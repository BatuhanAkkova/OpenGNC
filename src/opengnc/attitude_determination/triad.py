"""
TRIAD algorithm for attitude determination from two vectors.
"""

from __future__ import annotations

import numpy as np


def triad(
    v_body1: np.ndarray,
    v_body2: np.ndarray,
    v_ref1: np.ndarray,
    v_ref2: np.ndarray,
) -> np.ndarray:
    r"""
    Compute the DCM from reference to body frame using TRIAD.

    TRIAD (Tri-Axial Attitude Determination) construction:
    - $\mathbf{t}_{1b} = \mathbf{b}_1$
    - $\mathbf{t}_{2b} = \frac{\mathbf{b}_1 \times \mathbf{b}_2}{\|\mathbf{b}_1 \times \mathbf{b}_2\|}$
    - $\mathbf{t}_{3b} = \mathbf{t}_{1b} \times \mathbf{t}_{2b}$

    The Direction Cosine Matrix is:
    $\mathbf{R}_{BI} = [\mathbf{t}_{1b}\ \mathbf{t}_{2b}\ \mathbf{t}_{3b}] [\mathbf{t}_{1r}\ \mathbf{t}_{2r}\ \mathbf{t}_{3r}]^T$

    Parameters
    ----------
    v_body1 : np.ndarray
        Primary body vector (3,).
    v_body2 : np.ndarray
        Secondary body vector (3,).
    v_ref1 : np.ndarray
        Primary reference vector (3,).
    v_ref2 : np.ndarray
        Secondary reference vector (3,).

    Returns
    -------
    np.ndarray
        $3 \times 3$ DCM $\mathbf{R}_{BI}$ (Reference $\to$ Body).

    Raises
    ------
    ValueError
        If vectors are collinear or zero-length.
    """
    # Normalize inputs
    b1 = np.asarray(v_body1) / np.linalg.norm(v_body1)
    b2 = np.asarray(v_body2) / np.linalg.norm(v_body2)
    r1 = np.asarray(v_ref1) / np.linalg.norm(v_ref1)
    r2 = np.asarray(v_ref2) / np.linalg.norm(v_ref2)

    # Construct Body Triad
    t1b = b1
    t2b_raw = np.cross(b1, b2)
    norm_t2b = np.linalg.norm(t2b_raw)

    if norm_t2b < 1e-10:
        raise ValueError("Body vectors are collinear or nearly collinear.")

    t2b = t2b_raw / norm_t2b
    t3b = np.cross(t1b, t2b)

    # Construct Reference Triad
    t1r = r1
    t2r_raw = np.cross(r1, r2)
    norm_t2r = np.linalg.norm(t2r_raw)

    if norm_t2r < 1e-10:
        raise ValueError("Reference vectors are collinear or nearly collinear.")

    t2r = t2r_raw / norm_t2r
    t3r = np.cross(t1r, t2r)

    # Result: R_BI = [t1b t2b t3b] * [t1r t2r t3r]^T
    m_b = np.column_stack((t1b, t2b, t3b))
    m_r = np.column_stack((t1r, t2r, t3r))

    return np.asarray(m_b @ m_r.T)
