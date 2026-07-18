"""
Terrain-Relative Navigation (TRN) feature matching and localization update.
"""


import numpy as np


class FeatureMatchingTRN:
    """
    Terrain-Relative Navigation (TRN) Feature Matcher.

    Correlates observed landmarks (from sensors like LIDAR or cameras) with
    a known map database using efficient nearest-neighbor search.

    Parameters
    ----------
    map_database : List[np.ndarray]
        List of absolute landmark coordinates (m).
    """

    def __init__(self, map_database: list[np.ndarray]) -> None:
        """Initialize TRN map."""
        self.map = np.stack([np.asarray(m) for m in map_database])

    def match_features(
        self,
        observed_features: list[np.ndarray],
        dist_threshold: float = 10.0
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Match observed features to the global map.

        Parameters
        ----------
        observed_features : List[np.ndarray]
            Landmarks relative to the current vehicle state (m).
        dist_threshold : float, optional
            Max distance for correlation (m). Default is 10.0.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            Pairs of (Map Position, Observed Position).
        """
        matches = []
        for obs in observed_features:
            obs_v = np.asarray(obs)
            # Vectorized distance check
            dists = np.linalg.norm(self.map - obs_v, axis=1)
            idx = np.argmin(dists)

            if dists[idx] < dist_threshold:
                matches.append((self.map[idx], obs_v))
        return matches


def map_relative_localization_update(
    x_state: np.ndarray,
    p_cov: np.ndarray,
    matches: list[tuple[np.ndarray, np.ndarray]],
    r_noise: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    EKF Measurement Update using TRN feature matches.

    Corrects the global state estimate using residuals between map-known
    landmarks and their estimated positions.

    Parameters
    ----------
    x_state : np.ndarray
        Current filter state $[\mathbf{r}, ...]^T$.
    p_cov : np.ndarray
        Estimation covariance matrix.
    matches : List[Tuple[np.ndarray, np.ndarray]]
        Landmark correspondences from `match_features`.
    r_noise : np.ndarray
        Sensor noise covariance ($3\times 3$).

    Returns
    -------
    updated_x : np.ndarray
        Updated state estimate.
    updated_p : np.ndarray
        Updated covariance matrix.
    """
    if not matches:
        return x_state, p_cov

    x, p = x_state.copy(), p_cov.copy()
    rv = np.asarray(r_noise)

    for map_pos, obs_pos in matches:
        # Observation matrix H (Mapping global position to relative observation)
        h_mat = np.zeros((3, len(x)))
        h_mat[:, :3] = np.eye(3)

        resid = np.asarray(obs_pos) - h_mat @ x

        # Kalman Gain K = P H^T (H P H^T + R)^-1
        s_mat = h_mat @ p @ h_mat.T + rv
        k_gain = p @ h_mat.T @ np.linalg.inv(s_mat)

        x += k_gain @ resid
        p = (np.eye(len(x)) - k_gain @ h_mat) @ p

    return x, p




