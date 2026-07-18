"""
Davenport's q-method for optimal attitude determination (Wahba's problem).
"""


import numpy as np

from opengnc.utils.quat_utils import quat_normalize


def davenport_q(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Solve for the optimal attitude quaternion using Davenport's q-method.

    The optimal quaternion $\mathbf{q}$ is the eigenvector corresponding to the
    maximum eigenvalue $\lambda_{max}$ of the Davenport K-matrix:
    $\mathbf{K} \mathbf{q} = \lambda \mathbf{q}$

    Parameters
    ----------
    body_vectors : np.ndarray
        Measured body vectors (N, 3).
    ref_vectors : np.ndarray
        Reference inertial vectors (N, 3).
    weights : np.ndarray | None, optional
        Weights for observations (N,).

    Returns
    -------
    np.ndarray
        Optimal normalized quaternion $[x, y, z, w]$.

    Raises
    ------
    ValueError
        On dimension mismatch.
    """
    b_vecs = np.asarray(body_vectors)
    r_vecs = np.asarray(ref_vectors)

    if b_vecs.shape != r_vecs.shape:
        raise ValueError("Body and reference vector arrays must have the same shape.")

    n_vecs = b_vecs.shape[0]
    w = np.asarray(weights) if weights is not None else np.ones(n_vecs) / n_vecs

    if len(w) != n_vecs:
        raise ValueError("Number of weights must match number of vectors.")

    # Normalize vectors and compute profile matrix B
    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    b_matrix = np.zeros((3, 3))
    for i in range(n_vecs):
        b_matrix += w[i] * np.outer(b_norm[i], r_norm[i])

    # K-matrix components
    s_matrix = b_matrix + b_matrix.T
    sigma = float(np.trace(b_matrix))
    z_vec = np.array([
        b_matrix[2, 1] - b_matrix[1, 2],
        b_matrix[0, 2] - b_matrix[2, 0],
        b_matrix[1, 0] - b_matrix[0, 1]
    ])

    # Construct Davenport K-matrix
    k_matrix = np.zeros((4, 4))
    k_matrix[0:3, 0:3] = s_matrix - sigma * np.eye(3)
    k_matrix[0:3, 3] = z_vec
    k_matrix[3, 0:3] = z_vec
    k_matrix[3, 3] = sigma

    # The optimal quaternion is the eigenvector of K with largest eigenvalue
    vals, vecs = np.linalg.eigh(k_matrix)
    q_opt = vecs[:, np.argmax(vals)]

    return quat_normalize(q_opt)




