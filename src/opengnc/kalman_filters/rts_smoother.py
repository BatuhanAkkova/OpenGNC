"""
Rauch-Tung-Striebel (RTS) Smoother for linear systems.
"""

import numpy as np


def rts_smoother(
    x_filtered_list: list[np.ndarray],
    p_filtered_list: list[np.ndarray],
    f_mats: list[np.ndarray],
    q_mats: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel (RTS) Smoother for linear systems.

    Performs a backward pass over Kalman filter results to provide optimal
    minimum-variance estimates utilizing all future information (fixed-interval
    smoothing).

    Parameters
    ----------
    x_filtered_list : list[np.ndarray]
        List of filtered state estimates $x_{k|k}$ (N steps).
    p_filtered_list : list[np.ndarray]
        List of filtered covariances $P_{k|k}$ (N steps).
    f_mats : list[np.ndarray]
        List of state transition matrices $F_k$ from $k$ to $k+1$. (N-1 steps).
    q_mats : list[np.ndarray]
        List of process noise covariances $Q_k$ from $k$ to $k+1$. (N-1 steps).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_smoothed, p_smoothed) arrays.
        - x_smoothed: (N, dim_x)
        - p_smoothed: (N, dim_x, dim_x)
    """
    num_steps = len(x_filtered_list)
    dim_x = x_filtered_list[0].shape[0]

    x_smoothed = np.zeros((num_steps, dim_x))
    p_smoothed = np.zeros((num_steps, dim_x, dim_x))

    # 1. Initialize with terminal filtered state
    x_smoothed[-1] = x_filtered_list[-1]
    p_smoothed[-1] = p_filtered_list[-1]

    # 2. Sequential Backward Pass
    for k in range(num_steps - 2, -1, -1):
        f = f_mats[k]
        q = q_mats[k]
        xf = x_filtered_list[k]
        pf = p_filtered_list[k]

        # a. Predicted state and covariance at step k
        x_pred = f @ xf
        p_pred = (f @ pf @ f.T) + q

        # b. Smoother Gain: C_k = P_k|k * F_k' * [P_k+1|k]^-1
        gain_c = pf @ f.T @ np.linalg.inv(p_pred)

        # c. Smooth state and covariance
        x_smoothed[k] = xf + gain_c @ (x_smoothed[k + 1] - x_pred)
        p_smoothed[k] = pf + gain_c @ (p_smoothed[k + 1] - p_pred) @ gain_c.T

    return x_smoothed, p_smoothed




