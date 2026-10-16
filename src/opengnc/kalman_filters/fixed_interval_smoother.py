"""
Fixed-Interval Smoother (Fraser-Potter / Two-Filter) for linear systems.
"""

import numpy as np


def fixed_interval_smoother(
    x_forward: list[np.ndarray],
    p_forward: list[np.ndarray],
    f_mats: list[np.ndarray],
    q_mats: list[np.ndarray],
    z_meas: list[np.ndarray],
    h_mats: list[np.ndarray],
    r_mats: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-Interval Smoother (Two-Filter / Fraser-Potter approach).

    Combines a forward-running Kalman filter with a backward-running information
    filter to produce the optimal estimate at every point in a fixed interval.

    Parameters
    ----------
    x_forward : list of np.ndarray
        Forward filtered states $x_{k|k}$. Length N.
    p_forward : list of np.ndarray
        Forward filtered covariances $P_{k|k}$. Length N.
    f_mats : list of np.ndarray
        State transition matrices $F_k$ (from $k$ to $k+1$). Length N-1.
    q_mats : list of np.ndarray
        Process noise covariances $Q_k$ (from $k$ to $k+1$). Length N-1.
    z_meas : list of np.ndarray
        Measurements $z_k$. Length N.
    h_mats : list of np.ndarray
        Measurement matrices $H_k$. Length N.
    r_mats : list of np.ndarray
        Measurement noise covariances $R_k$. Length N.

    Returns
    -------
    x_smooth : np.ndarray
        Smoothed states.
    p_smooth : np.ndarray
        Smoothed covariances.
    """
    num_steps = len(x_forward)
    dim_x = x_forward[0].shape[0]

    # Backward Information Filter
    # y_b: information state (Y * x)
    # Y_b: information matrix (inv(P))
    y_info = np.zeros((num_steps, dim_x))
    y_mat_info = np.zeros((num_steps, dim_x, dim_x))

    # Backward pass
    for k in range(num_steps - 2, -1, -1):
        # Update with measurement at k+1
        h_mat = h_mats[k + 1]
        r_inv = np.linalg.inv(r_mats[k + 1])

        # Information update
        y_hat_mat = y_mat_info[k + 1] + h_mat.T @ r_inv @ h_mat
        y_hat_vec = y_info[k + 1] + h_mat.T @ r_inv @ z_meas[k + 1]

        # Backward predict to k
        f_mat = f_mats[k]
        q_mat = q_mats[k]

        # Backward prediction (Information form)
        eye_x = np.eye(dim_x)
        tmp_inv = np.linalg.inv(eye_x + y_hat_mat @ q_mat)
        y_mat_info[k] = f_mat.T @ y_hat_mat @ tmp_inv @ f_mat
        y_info[k] = f_mat.T @ tmp_inv @ y_hat_vec

    # Combine Forward and Backward
    x_smoothed = np.zeros((num_steps, dim_x))
    p_smoothed = np.zeros((num_steps, dim_x, dim_x))

    for k in range(num_steps):
        p_f_inv = np.linalg.inv(p_forward[k])
        p_smoothed[k] = np.linalg.inv(p_f_inv + y_mat_info[k])
        x_smoothed[k] = p_smoothed[k] @ (p_f_inv @ x_forward[k] + y_info[k])

    return x_smoothed, p_smoothed
