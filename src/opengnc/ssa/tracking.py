"""
Object Tracking and Orbit Correlation Utilities.
"""

import numpy as np


def compute_mahalanobis_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray
) -> float:
    r"""
    Compute the Mahalanobis statistical distance between two state estimates.

    Used to measure the separation between two Gaussian distributions
    accounting for their covariance.
    Formula: $d_M = \sqrt{(\mathbf{x}_1 - \mathbf{x}_2)^T (\mathbf{P}_1 + \mathbf{P}_2)^{-1} (\mathbf{x}_1 - \mathbf{x}_2)}$.

    Parameters
    ----------
    x1, x2 : np.ndarray
        State vectors (e.g., Position/Velocity) (6,).
    cov1, cov2 : np.ndarray
        Estimation error covariance matrices (6, 6).

    Returns
    -------
    float
        Mahalanobis distance.
    """
    dx = np.asarray(x1) - np.asarray(x2)
    p_comb = np.asarray(cov1) + np.asarray(cov2)

    try:
        # Distance squared: d^2 = dx' * P^-1 * dx
        d2 = dx.T @ np.linalg.inv(p_comb) @ dx
        return float(np.sqrt(max(0, d2)))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean if singular
        return float(np.linalg.norm(dx))


def correlate_tracks(
    x1: np.ndarray,
    x2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    threshold: float = 3.0
) -> bool:
    """
    Correlate two tracks based on a Mahalanobis distance threshold.

    Used for track-to-track association in SSA catalogs.

    Parameters
    ----------
    x1, x2 : np.ndarray
        State vectors to compare.
    cov1, cov2 : np.ndarray
        State covariances.
    threshold : float, optional
        Correlation limit (n-sigma). Default is 3.0 (3-sigma).

    Returns
    -------
    bool
        True if the distance is within the specified threshold.
    """
    dist = compute_mahalanobis_distance(x1, x2, cov1, cov2)
    return dist <= threshold




