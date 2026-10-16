"""
QUEST algorithm for optimal attitude determination (Wahba's problem).
"""

import numpy as np

from opengnc.utils.quat_utils import quat_normalize


def quest(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> np.ndarray:
    r"""
    Solve for the optimal attitude quaternion using QUEST.

    QUEST (QUaternion ESTimator) minimizes Wahba's Loss:
    $J(\mathbf{R}) = \frac{1}{2} \sum_{i=1}^N w_i \|\mathbf{b}_i - \mathbf{R} \mathbf{r}_i\|^2$

    It achieves this by finding the maximum eigenvalue of the K-matrix:
    $\mathbf{K} \mathbf{q}_{opt} = \lambda_{max} \mathbf{q}_{opt}$

    Parameters
    ----------
    body_vectors : np.ndarray
        Body-frame measurements (N, 3).
    ref_vectors : np.ndarray
        Inertial-frame references (N, 3).
    weights : np.ndarray | None, optional
        Weights for observations (N,).
    tol : float, optional
        Newton convergence tolerance. Default 1e-12.
    max_iter : int, optional
        Max iterations. Default 20.

    Returns
    -------
    np.ndarray
        Hamilton quaternion $[x, y, z, w]$ (Inertial $\to$ Body).

    Raises
    ------
    ValueError
        On dimension mismatch.
    """
    b_vecs = np.asarray(body_vectors)
    r_vecs = np.asarray(ref_vectors)

    n_vecs = b_vecs.shape[0]
    if r_vecs.shape[0] != n_vecs:
        raise ValueError("Body and reference vector count mismatch.")

    w = np.asarray(weights) if weights is not None else np.ones(n_vecs) / n_vecs
    if len(w) != n_vecs:
        raise ValueError("Weight vector dimension mismatch.")

    # Normalize vectors and compute B matrix
    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    b_matrix = np.zeros((3, 3))
    for i in range(n_vecs):
        b_matrix += w[i] * np.outer(b_norm[i], r_norm[i])

    # K-matrix components
    s_matrix = b_matrix + b_matrix.T
    sigma = float(np.trace(b_matrix))
    z_vec = np.array(
        [
            b_matrix[2, 1] - b_matrix[1, 2],
            b_matrix[0, 2] - b_matrix[2, 0],
            b_matrix[1, 0] - b_matrix[0, 1],
        ]
    )

    # Characteristic polynomial coefficients for lambda search
    b_frob_sq = float(np.trace(b_matrix @ b_matrix.T))
    adj_b = np.array(
        [
            [
                b_matrix[1, 1] * b_matrix[2, 2] - b_matrix[1, 2] * b_matrix[2, 1],
                b_matrix[0, 2] * b_matrix[2, 1] - b_matrix[0, 1] * b_matrix[2, 2],
                b_matrix[0, 1] * b_matrix[1, 2] - b_matrix[0, 2] * b_matrix[1, 1],
            ],
            [
                b_matrix[1, 2] * b_matrix[2, 0] - b_matrix[1, 0] * b_matrix[2, 2],
                b_matrix[0, 0] * b_matrix[2, 2] - b_matrix[0, 2] * b_matrix[2, 0],
                b_matrix[0, 2] * b_matrix[1, 0] - b_matrix[0, 0] * b_matrix[1, 2],
            ],
            [
                b_matrix[1, 0] * b_matrix[2, 1] - b_matrix[1, 1] * b_matrix[2, 0],
                b_matrix[0, 1] * b_matrix[2, 0] - b_matrix[0, 0] * b_matrix[2, 1],
                b_matrix[0, 0] * b_matrix[1, 1] - b_matrix[0, 1] * b_matrix[1, 0],
            ],
        ]
    ).T  # Adjugate is the transpose of the cofactor matrix

    adj_b_frob_sq = float(np.trace(adj_b @ adj_b.T))
    det_b = float(np.linalg.det(b_matrix))

    # Solve for lambda_max using Newton-Raphson starting at sum of weights
    lam = float(np.sum(w))
    for _ in range(max_iter):
        # f(lam) and f'(lam) from Davenport characteristic equation
        f_val = (lam**2 - b_frob_sq) ** 2 - 8 * lam * det_b - 4 * adj_b_frob_sq
        fp_val = 4 * lam * (lam**2 - b_frob_sq) - 8 * det_b

        delta = f_val / fp_val
        lam -= delta
        if abs(delta) < tol:
            break

    # Compute optimal quaternion via Gibbs vector p = ( (l+s)I - S )^-1 z
    m_inv_lhs = (lam + sigma) * np.eye(3) - s_matrix
    p_vec = np.linalg.solve(m_inv_lhs, z_vec)

    q_opt = np.array([p_vec[0], p_vec[1], p_vec[2], 1.0])
    return quat_normalize(q_opt)
