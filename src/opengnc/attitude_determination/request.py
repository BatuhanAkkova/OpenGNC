"""
Recursive QUEST (REQUEST) algorithm for recursive attitude estimation.
"""


from typing import Any

import numpy as np

from opengnc.utils.quat_utils import quat_normalize


class RequestFilter:
    r"""
    Recursive QUEST (REQUEST) filter.

    Enables recursive attitude estimation by updating the Davenport
    K-matrix with a fading memory factor $\rho$:
    $\mathbf{K}_k = \rho \mathbf{K}_{k-1} + \delta \mathbf{K}_k$

    Parameters
    ----------
    initial_k : np.ndarray | None, optional
        Initial $4 \times 4$ K-matrix. Default zero.
    """

    def __init__(self, initial_k: np.ndarray | None = None, **kwargs: Any) -> None:
        """Initialize filter state."""
        # Support both 'initial_k' and 'initial_K'
        k_val = initial_k if initial_k is not None else kwargs.get("initial_K")

        if k_val is not None:
            self.k = np.asarray(k_val, dtype=float)
        else:
            self.k = np.zeros((4, 4))

    @property
    def K(self) -> np.ndarray:
        r"""Alias for the accumulated $4 \times 4$ K-matrix."""
        return self.k

    @K.setter
    def K(self, value: np.ndarray) -> None:
        self.k = np.asarray(value)

    def update(
        self,
        body_vectors: np.ndarray,
        ref_vectors: np.ndarray,
        weights: np.ndarray | None = None,
        rho: float = 1.0,
    ) -> np.ndarray:
        r"""
        Update the accumulated K-matrix with new vector observations.

        Parameters
        ----------
        body_vectors : np.ndarray
            New measurements in the body frame (N, 3).
        ref_vectors : np.ndarray
            Corresponding reference vectors in the inertial frame (N, 3).
        weights : np.ndarray, optional
            Weights for the new measurements. Defaults to $1/N$.
        rho : float, optional
            Fading memory factor $0 < \\rho \\le 1$. Default is 1.0 (no fading).

        Returns
        -------
        np.ndarray
            The updated $4\\times 4$ K-matrix.
        """
        b_vecs = np.asarray(body_vectors)
        r_vecs = np.asarray(ref_vectors)
        n_vecs = b_vecs.shape[0]

        w = np.asarray(weights) if weights is not None else np.ones(n_vecs) / n_vecs

        # Normalize and compute incremental profile matrix dB
        b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
        r_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

        db_matrix = np.zeros((3, 3))
        for i in range(n_vecs):
            db_matrix += w[i] * np.outer(b_norm[i], r_norm[i])

        ds_matrix = db_matrix + db_matrix.T
        d_sigma = float(np.trace(db_matrix))
        dz_vec = np.array([
            db_matrix[2, 1] - db_matrix[1, 2],
            db_matrix[0, 2] - db_matrix[2, 0],
            db_matrix[1, 0] - db_matrix[0, 1]
        ])

        # Form incremental dK matrix
        dk_matrix = np.zeros((4, 4))
        dk_matrix[0:3, 0:3] = ds_matrix - d_sigma * np.eye(3)
        dk_matrix[0:3, 3] = dz_vec
        dk_matrix[3, 0:3] = dz_vec
        dk_matrix[3, 3] = d_sigma

        # Update rule: K = rho * K + dK
        self.k = rho * self.k + dk_matrix
        return self.k

    def get_quaternion(self) -> np.ndarray:
        """
        Extract the optimal normalized quaternion from the current K-matrix.

        Returns
        -------
        np.ndarray
            Optimal quaternion $[x, y, z, w]$ (Inertial -> Body).
        """
        vals, vecs = np.linalg.eigh(self.k)
        q_opt = vecs[:, np.argmax(vals)]
        return quat_normalize(q_opt)


def request(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: np.ndarray | None = None,
    initial_k: np.ndarray | None = None,
    rho: float = 1.0,
) -> np.ndarray:
    """
    One-shot REQUEST update helper.

    Parameters
    ----------
    body_vectors : np.ndarray
        New measurements in the body frame (N, 3).
    ref_vectors : np.ndarray
        Reference vectors in the inertial frame (N, 3).
    weights : np.ndarray, optional
        Weights for new measurements.
    initial_k : np.ndarray, optional
        Starting K-matrix.
    rho : float, optional
        Fading memory factor.

    Returns
    -------
    np.ndarray
        Optimal quaternion $[x, y, z, w]$.
    """
    rf = RequestFilter(initial_k)
    rf.update(body_vectors, ref_vectors, weights, rho)
    return rf.get_quaternion()




