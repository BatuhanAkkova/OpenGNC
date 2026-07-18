"""
Conjunction analysis and probability of collision utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar


def compute_pc_foster(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
) -> float:
    r"""
    Compute probability of collision via Foster's method.

    The encounter is projected into a 2D collision plane and the Gaussian
    probability density is integrated over the circular hard-body region.

    Parameters
    ----------
    r1, v1 : np.ndarray
        ECI state of object 1 at TCA in meters and meters per second.
    cov1 : np.ndarray
        ``3 x 3`` covariance of object 1 in square meters.
    r2, v2 : np.ndarray
        ECI state of object 2 at TCA in meters and meters per second.
    cov2 : np.ndarray
        ``3 x 3`` covariance of object 2 in square meters.
    hbr : float
        Combined hard-body radius in meters.

    Returns
    -------
    float
        Probability of collision in ``[0, 1]``.
    """
    rv1, rv2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    vv1, vv2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    cv1, cv2 = np.asarray(cov1, dtype=float), np.asarray(cov2, dtype=float)

    r_rel = rv1 - rv2
    v_rel = vv1 - vv2
    v_mag = np.linalg.norm(v_rel)

    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small for encounter projection.")

    combined_cov = cv1 + cv2
    z_hat = v_rel / v_mag

    if np.linalg.norm(r_rel) > 1e-4:
        x_hat = r_rel - np.dot(r_rel, z_hat) * z_hat
        x_mag = np.linalg.norm(x_hat)
        if x_mag > 1e-6:
            x_hat /= x_mag
        else:
            x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_hat -= np.dot(x_hat, z_hat) * z_hat
            x_hat /= np.linalg.norm(x_hat)
    else:
        x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_hat -= np.dot(x_hat, z_hat) * z_hat
        x_hat /= np.linalg.norm(x_hat)

    y_hat = np.cross(z_hat, x_hat)
    m_rot = np.vstack([x_hat, y_hat, z_hat])

    r_enc = m_rot @ r_rel
    cov_enc = m_rot @ combined_cov @ m_rot.T
    cov_2d = cov_enc[:2, :2]
    det_c = np.linalg.det(cov_2d)

    if det_c < 1e-15:
        return 0.0

    inv_c = np.linalg.inv(cov_2d)
    x_c, y_c = float(r_enc[0]), float(r_enc[1])

    def pdf_2d(y_val: float, x_val: float) -> float:
        delta = np.array([x_val - x_c, y_val - y_c], dtype=float)
        arg = -0.5 * delta.T @ inv_c @ delta
        return float((1.0 / (2.0 * np.pi * np.sqrt(det_c))) * np.exp(arg))

    pc, _ = dblquad(
        pdf_2d,
        -hbr,
        hbr,
        lambda y_val: -np.sqrt(max(0.0, hbr**2 - y_val**2)),
        lambda y_val: np.sqrt(max(0.0, hbr**2 - y_val**2)),
    )
    return float(pc)


def compute_pc_chan(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
) -> float:
    r"""
    Compute probability of collision via Chan's analytical approximation.

    Parameters
    ----------
    r1, r2 : np.ndarray
        ECI position vectors at TCA in meters.
    v1, v2 : np.ndarray
        ECI velocity vectors at TCA in meters per second.
    cov1, cov2 : np.ndarray
        ``3 x 3`` covariance matrices in square meters.
    hbr : float
        Combined hard-body radius in meters.

    Returns
    -------
    float
        Computed probability of collision.
    """
    rv1, rv2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    vv1, vv2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    cv1, cv2 = np.asarray(cov1, dtype=float), np.asarray(cov2, dtype=float)

    r_rel = rv1 - rv2
    v_rel = vv1 - vv2
    v_mag = np.linalg.norm(v_rel)

    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small.")

    z_hat = v_rel / v_mag
    x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_hat -= np.dot(x_hat, z_hat) * z_hat
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    m_rot = np.vstack([x_hat, y_hat, z_hat])

    r_enc = m_rot @ r_rel
    cov_2d = (m_rot @ (cv1 + cv2) @ m_rot.T)[:2, :2]

    vals, vecs = np.linalg.eigh(cov_2d)
    if np.any(vals <= 0):
        return 0.0

    r_p = vecs.T @ r_enc[:2]
    sig_x, sig_y = np.sqrt(vals[0]), np.sqrt(vals[1])

    u_val = (r_p[0] ** 2 / vals[0]) + (r_p[1] ** 2 / vals[1])
    v_val = hbr**2 / (sig_x * sig_y)

    pc = 0.0
    term_u = np.exp(-u_val / 2.0)
    term_v = np.exp(-v_val / 2.0)
    sum_v = term_v

    for n_idx in range(50):
        inc = term_u * (1.0 - sum_v)
        pc += inc
        if inc < 1e-15:
            break
        term_u *= (u_val / 2.0) / (n_idx + 1)
        term_v *= (v_val / 2.0) / (n_idx + 1)
        sum_v += term_v

    return float(pc)


def propagate_covariance(
    cov0: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Linearly propagate a ``3 x 3`` position covariance using a simple growth model.
    """
    mu = 3.986004418e14
    r_mag = np.linalg.norm(r)
    n_val = np.sqrt(mu / r_mag**3)
    q_drift = 1e-4 * dt**2 * np.eye(3)
    spread_factor = 1.0 + (0.01 * n_val * abs(dt))
    return np.asarray(cov0 * (spread_factor**2) + q_drift, dtype=float)


def find_tca(
    r1_func: Callable[[float], np.ndarray],
    r2_func: Callable[[float], np.ndarray],
    t_start: float,
    t_end: float,
    tol: float = 0.1,
) -> float:
    """
    Find the time of closest approach between two objects.

    Parameters
    ----------
    r1_func, r2_func : Callable[[float], np.ndarray]
        Functions returning ECI position at time ``t``.
    t_start, t_end : float
        Search window in seconds.
    tol : float, optional
        Scalar search tolerance in seconds.

    Returns
    -------
    float
        Time of closest approach.
    """

    def dist_sq(t_val: float) -> float:
        dr = np.asarray(r1_func(t_val), dtype=float) - np.asarray(r2_func(t_val), dtype=float)
        return float(np.dot(dr, dr))

    res = minimize_scalar(
        dist_sq,
        bounds=(t_start, t_end),
        method="bounded",
        options={"xatol": tol},
    )
    return float(cast(float, res.x))
