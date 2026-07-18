"""
Fast Optimal Attitude Matrix (FOAM) algorithm for attitude determination.
"""

from typing import cast

import numpy as np


def foam(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> np.ndarray:
    r"""
    Solve for the optimal attitude matrix using FOAM.

    Parameters
    ----------
    body_vectors : np.ndarray
        Body measurements with shape ``(N, 3)``.
    ref_vectors : np.ndarray
        Inertial reference vectors with shape ``(N, 3)``.
    weights : np.ndarray | None, optional
        Optional weights of shape ``(N,)``.
    tol : float, optional
        Newton iteration tolerance.
    max_iter : int, optional
        Maximum Newton iterations.

    Returns
    -------
    np.ndarray
        Optimal ``3 x 3`` direction cosine matrix.
    """
    b_vecs = np.asarray(body_vectors, dtype=float)
    r_vecs = np.asarray(ref_vectors, dtype=float)

    if b_vecs.shape != r_vecs.shape:
        raise ValueError("Body and reference vector arrays must have the same shape.")

    n_vecs = b_vecs.shape[0]
    w_vec = np.asarray(weights, dtype=float) if weights is not None else np.ones(n_vecs) / n_vecs
    if len(w_vec) != n_vecs:
        raise ValueError("Number of weights must match number of vectors.")

    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    b_matrix = np.zeros((3, 3), dtype=float)
    for i in range(n_vecs):
        b_matrix += w_vec[i] * np.outer(b_norm[i], r_norm[i])

    det_b = float(np.linalg.det(b_matrix))
    adj_b = np.zeros((3, 3), dtype=float)
    adj_b[0, 0] = b_matrix[1, 1] * b_matrix[2, 2] - b_matrix[1, 2] * b_matrix[2, 1]
    adj_b[0, 1] = b_matrix[0, 2] * b_matrix[2, 1] - b_matrix[0, 1] * b_matrix[2, 2]
    adj_b[0, 2] = b_matrix[0, 1] * b_matrix[1, 2] - b_matrix[0, 2] * b_matrix[1, 1]
    adj_b[1, 0] = b_matrix[1, 2] * b_matrix[2, 0] - b_matrix[1, 0] * b_matrix[2, 2]
    adj_b[1, 1] = b_matrix[0, 0] * b_matrix[2, 2] - b_matrix[0, 2] * b_matrix[2, 0]
    adj_b[1, 2] = b_matrix[0, 2] * b_matrix[1, 0] - b_matrix[0, 0] * b_matrix[1, 2]
    adj_b[2, 0] = b_matrix[1, 0] * b_matrix[2, 1] - b_matrix[1, 1] * b_matrix[2, 0]
    adj_b[2, 1] = b_matrix[0, 1] * b_matrix[2, 0] - b_matrix[0, 0] * b_matrix[2, 1]
    adj_b[2, 2] = b_matrix[0, 0] * b_matrix[1, 1] - b_matrix[0, 1] * b_matrix[1, 0]

    b_frob_sq = float(np.trace(b_matrix @ b_matrix.T))
    adj_b_frob_sq = float(np.trace(adj_b @ adj_b.T))

    lam = float(np.sum(w_vec))
    for _ in range(max_iter):
        f_val = (lam**2 - b_frob_sq) ** 2 - 8.0 * lam * det_b - 4.0 * adj_b_frob_sq
        fp_val = 4.0 * lam * (lam**2 - b_frob_sq) - 8.0 * det_b
        delta = f_val / fp_val
        lam -= delta
        if abs(delta) < tol:
            break

    num = (
        (lam**2 + b_frob_sq) * b_matrix
        + 2.0 * lam * adj_b.T
        - 2.0 * (b_matrix @ b_matrix.T @ b_matrix)
    )
    den = lam * (lam**2 - b_frob_sq) - 2.0 * det_b
    return cast(np.ndarray, np.asarray(num / den, dtype=float))
