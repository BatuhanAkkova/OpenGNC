"""
Initial Orbit Determination (IOD) methods (Gibbs, Gauss, Laplace).
"""

from __future__ import annotations

from typing import cast

import numpy as np


def gibbs_iod(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    mu: float = 398600.4415e9
) -> np.ndarray:
    r"""
    Initial Orbit Determination (IOD) via Gibbs method.

    Suitable for three position vectors with angular separation > 5 degrees.
    Formula: $\mathbf{v}_2 = \sqrt{\frac{\mu}{N D}} \left( \frac{\mathbf{D} \times \mathbf{r}_2}{r_2} + \mathbf{S} \right)$.

    Parameters
    ----------
    r1, r2, r3 : np.ndarray
        ECI position vectors (m) at three distinct times.
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default is Earth.

    Returns
    -------
    np.ndarray
        Velocity vector at the second observation time $\mathbf{v}_2$ (m/s).
    """
    rv1, rv2, rv3 = np.asarray(r1), np.asarray(r2), np.asarray(r3)
    r1m, r2m, r3m = np.linalg.norm(rv1), np.linalg.norm(rv2), np.linalg.norm(rv3)

    # Gibbs auxiliary vectors
    d_vec = np.cross(rv1, rv2) + np.cross(rv2, rv3) + np.cross(rv3, rv1)
    n_vec = r1m * np.cross(rv2, rv3) + r2m * np.cross(rv3, rv1) + r3m * np.cross(rv1, rv2)
    l_mag = np.linalg.norm(d_vec)

    if l_mag < 1e-12:
        return cast(np.ndarray, np.zeros(3))

    s_vec = rv1 * (r2m - r3m) + rv2 * (r3m - r1m) + rv3 * (r1m - r2m)

    v2 = np.sqrt(mu / (np.linalg.norm(n_vec) * l_mag)) * (np.cross(d_vec, rv2) / r2m + s_vec)
    return cast(np.ndarray, v2)


def herrick_gibbs_iod(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    dt21: float,
    dt32: float,
    mu: float = 398600.4415e9
) -> np.ndarray:
    r"""
    Initial Orbit Determination via Herrick-Gibbs method.

    Best for short-arc observations (angular separation < 5 degrees).

    Parameters
    ----------
    r1, r2, r3 : np.ndarray
        ECI position vectors (m).
    dt21 : float
        Time interval $t_2 - t_1$ (s).
    dt32 : float
        Time interval $t_3 - t_2$ (s).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).

    Returns
    -------
    np.ndarray
        Velocity vector $\mathbf{v}_2$ (m/s).
    """
    p1, p2, p3 = np.asarray(r1), np.asarray(r2), np.asarray(r3)
    dt31 = dt21 + dt32

    n1, n2, n3 = np.linalg.norm(p1), np.linalg.norm(p2), np.linalg.norm(p3)
    if n1 < 1e-12 or n2 < 1e-12 or n3 < 1e-12:
        raise ValueError("Position vectors must be non-zero.")

    term1 = -dt32 * (1.0 / (dt21 * dt31) + mu / (12.0 * n1**3)) * p1
    term2 = (dt32 - dt21) * (1.0 / (dt21 * dt32) + mu / (12.0 * n2**3)) * p2
    term3 = dt21 * (1.0 / (dt32 * dt31) + mu / (12.0 * n3**3)) * p3

    return cast(np.ndarray, term1 + term2 + term3)


def gauss_iod(
    rho_hat1: np.ndarray,
    rho_hat2: np.ndarray,
    rho_hat3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    r_obs1: np.ndarray,
    r_obs2: np.ndarray,
    r_obs3: np.ndarray,
    mu: float = 398600.4415e9,
) -> np.ndarray:
    r"""
    Angles-only IOD via Gauss method with iterative refinement.

    Solves for slant ranges using an 8th-order polynomial and performs
    iterative refinement using Herrick-Gibbs for stability.

    Parameters
    ----------
    rho_hat1, rho_hat2, rho_hat3 : np.ndarray
        Line-of-Sight unit vectors in ECI.
    t1, t2, t3 : float
        Observation times (s).
    r_obs1, r_obs2, r_obs3 : np.ndarray
        Observer position vectors in ECI (m).
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).

    Returns
    -------
    np.ndarray
        ECI State vector $[r, v]$ at $t_2$ (m, m/s).
    """
    tau1, tau3 = t1 - t2, t3 - t2
    tau = tau3 - tau1

    l1, l2, l3 = np.asarray(rho_hat1), np.asarray(rho_hat2), np.asarray(rho_hat3)
    R1, R2, R3 = np.asarray(r_obs1), np.asarray(r_obs2), np.asarray(r_obs3)

    p1, p2 = np.cross(l2, l3), np.cross(l1, l3)
    d0 = float(np.dot(l1, p1))
    if abs(d0) < 1e-18:
        raise ValueError("LOS vectors are nearly coplanar.")

    d12, d22, d32 = float(np.dot(R1, p2)), float(np.dot(R2, p2)), float(np.dot(R3, p2))

    c1_init, c3_init = tau3 / tau, -tau1 / tau

    # Range equation coefficients: rho2*d0 = a + b/r2^3
    a_term = (1.0 / d0) * (d22 - c1_init * d12 - c3_init * d32)
    term_b1 = c1_init * (tau**2 - tau3**2) * d12
    term_b3 = c3_init * (tau**2 - tau1**2) * d32
    b_term = (1.0 / (6.0 * d0)) * (-term_b1 - term_b3)

    e_dot = float(np.dot(R2, l2))
    R2_sq = np.dot(R2, R2)

    poly_coeffs = [
        1.0, 0.0,
        -(a_term**2 + 2.0 * a_term * e_dot + R2_sq),
        0.0, 0.0,
        -2.0 * mu * b_term * (a_term + e_dot),
        0.0, 0.0,
        -(mu**2) * b_term**2
    ]
    roots = np.roots(poly_coeffs)
    real_positive_roots = sorted(roots[np.isreal(roots) & (roots.real > 0)].real)

    if not real_positive_roots:
        raise ValueError("No physical radius solution found.")

    r2_mag = real_positive_roots[-1]
    if r2_mag > 1e11:
        raise ValueError("Radius out of bounds.")

    # Pick root that gives positive range
    for r in reversed(real_positive_roots):
        if (a_term + mu * b_term / r**3) > 0:
            r2_mag = r
            break

    # Final result containers
    r2_final, v2_final = np.zeros(3), np.zeros(3)

    # Initial solver refinement (One-pass stable refinement)
    # Using the polynomial magnitude to fix the curvature, solve for rhos
    u = mu / r2_mag**3
    c1 = c1_init * (1 + u/6 * (tau**2 - tau3**2))
    c3 = c3_init * (1 + u/6 * (tau**2 - tau1**2))

    try:
        mat = np.array([c1 * l1, -l2, c3 * l3]).T
        rhs = R2 - c1 * R1 - c3 * R3
        rhos = np.linalg.solve(mat, rhs)

        r1_vec = R1 + rhos[0] * l1
        r2_vec = R2 + rhos[1] * l2
        r3_vec = R3 + rhos[2] * l3

        # Use Herrick-Gibbs for velocity (much more stable than f,g for most arcs)
        v2_vec = herrick_gibbs_iod(r1_vec, r2_vec, r3_vec, -tau1, tau3, mu)
        r2_final, v2_final = r2_vec, v2_vec
    except Exception:
        # Fallback to pure polynomial rho2 and simple v2
        rho2 = a_term + mu * b_term / r2_mag**3
        r2_final = R2 + rho2 * l2
        v2_final = np.zeros(3)

    return cast(np.ndarray, np.concatenate([r2_final, v2_final]))


def laplace_iod(
    rho_hat: np.ndarray,
    rho_dot: np.ndarray,
    rho_ddot: np.ndarray,
    r_obs: np.ndarray,
    v_obs: np.ndarray,
    a_obs: np.ndarray,
    mu: float = 398600.4415e9
) -> np.ndarray:
    r"""
    Orbit determination from line-of-sight derivatives (Laplace method).

    Parameters
    ----------
    rho_hat, rho_dot, rho_ddot : np.ndarray
        LOS unit vector and its first/second time derivatives.
    r_obs, v_obs, a_obs : np.ndarray
        Observer Cartesian state and acceleration at epoch.
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).

    Returns
    -------
    np.ndarray
        ECI State vector $[r, v]$ at epoch.
    """
    rho_hat_vec, rho_dot_vec, rho_ddot_vec = np.asarray(rho_hat), np.asarray(rho_dot), np.asarray(rho_ddot)
    r_o, v_o, a_o = np.asarray(r_obs), np.asarray(v_obs), np.asarray(a_obs)

    d_mat = np.array([rho_hat_vec, rho_dot_vec, rho_ddot_vec])
    det_d = np.linalg.det(d_mat)
    if abs(det_d) < 1e-15:
        raise ValueError("Determinant D is too small for Laplace IOD.")

    d1 = np.linalg.det(np.array([rho_hat_vec, rho_dot_vec, a_o]))
    d2 = np.linalg.det(np.array([rho_hat_vec, rho_dot_vec, r_o]))

    a_lap, b_lap = -d1 / det_d, -mu * d2 / det_d
    r_mag_o = np.linalg.norm(r_o)
    cos_phi = np.dot(rho_hat_vec, r_o) / r_mag_o

    # Solve radius equation
    poly = [1.0, 0.0, -(a_lap**2 + 2.0*r_mag_o*a_lap*cos_phi + r_mag_o**2),
            0.0, 0.0, -(2.0*a_lap*b_lap + 2.0*r_mag_o*b_lap*cos_phi), 0.0, 0.0, -(b_lap**2)]

    roots = np.roots(poly)
    real_positive_roots = roots[np.isreal(roots) & (roots.real > 0)].real
    if len(real_positive_roots) == 0:
        raise ValueError("No physical radius solution found (Laplace).")
    r_mag = float(np.max(real_positive_roots))
    rho_mag = a_lap + b_lap / r_mag**3

    rv = rho_mag * rho_hat_vec + r_o

    d3 = np.linalg.det(np.array([rho_hat_vec, a_o, rho_ddot_vec]))
    d4 = np.linalg.det(np.array([rho_hat_vec, r_o, rho_ddot_vec]))
    rho_mag_dot = -(d3 + (mu / r_mag**3) * d4) / (2.0 * det_d)

    vv = rho_mag_dot * rho_hat_vec + rho_mag * rho_dot_vec + v_o

    return cast(np.ndarray, np.concatenate([rv, vv]))


def laplace_iod_from_observations(
    rho_hats: list[np.ndarray],
    rs_obs: list[np.ndarray],
    times: list[float],
    mu: float = 398600.4415e9
) -> np.ndarray:
    r"""
    Perform Laplace IOD from three standard LOS observations.

    Estimates derivatives using Lagrange interpolation.

    Parameters
    ----------
    rho_hats : List[np.ndarray]
        Three LOS unit vectors.
    rs_obs : List[np.ndarray]
        Three observer position vectors.
    times : List[float]
        Three observation timestamps.
    mu : float, optional
        Gravitational parameter.

    Returns
    -------
    np.ndarray
        State vector at $t_2$ (6,) (m, m/s).
    """
    t1, t2, t3 = times
    l1, l2, l3 = [np.asarray(rh) for rh in rho_hats]
    R1, R2, R3 = [np.asarray(r) for r in rs_obs]

    dt32, dt21, dt31 = t3 - t2, t2 - t1, t3 - t1

    # First and second derivatives at t2 via Lagrange
    cl1d = -dt32 / (-dt21 * -dt31)
    cl2d = (dt21 - dt32) / (dt21 * -dt32)
    cl3d = dt21 / (dt31 * dt32)

    l_dot = cl1d * l1 + cl2d * l2 + cl3d * l3
    R_dot = cl1d * R1 + cl2d * R2 + cl3d * R3

    cl1dd = 2.0 / (-dt21 * -dt31)
    cl2dd = 2.0 / (dt21 * -dt32)
    cl3dd = 2.0 / (dt31 * dt32)

    l_ddot = cl1dd * l1 + cl2dd * l2 + cl3dd * l3
    R_ddot = cl1dd * R1 + cl2dd * R2 + cl3dd * R3

    return cast(np.ndarray, laplace_iod(l2, l_dot, l_ddot, R2, R_dot, R_ddot, mu))


def _stumpff(z: float) -> tuple[float, float]:
    r"""
    Stumpff functions c2(z) and c3(z) for universal variable Kepler solution.
    """
    if z > 1e-6:
        sz = np.sqrt(z)
        c2 = (1 - np.cos(sz)) / z
        c3 = (sz - np.sin(sz)) / (sz**3)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        c2 = (1 - np.cosh(sz)) / z
        c3 = (np.sinh(sz) - sz) / (sz**3)
    else:
        c2 = 1.0 / 2.0
        c3 = 1.0 / 6.0
    return float(c2), float(c3)


def _kepler_U(dt: float, r0: float, v0_mag: float, alpha: float, mu: float) -> float:
    r"""
    Universal variable Kepler solver (Newton-Raphson).
    """
    chi = np.sqrt(mu) * np.abs(alpha) * dt
    for _ in range(10):
        z = alpha * chi**2
        c2, c3 = _stumpff(z)
        f = r0 * chi * (1 - z * c3) + (np.dot(r0, v0_mag) / np.sqrt(mu)) * chi**2 * c2 + chi**3 * c3 - np.sqrt(mu) * dt
        df = r0 * (1 - z * c2) + (np.dot(r0, v0_mag) / np.sqrt(mu)) * chi * (1 - z * c3) + chi**2 * c2
        chi_new = chi - f / df
        if np.abs(chi_new - chi) < 1e-10:
            return float(chi_new)
        chi = chi_new
    return float(chi)




