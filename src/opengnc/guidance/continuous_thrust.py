"""
Continuous thrust guidance laws including Q-law and ZEM/ZEV feedback.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize

from opengnc.utils.state_to_elements import eci2kepler


def q_law_guidance(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    target_oe: np.ndarray,
    mu: float,
    accel_max: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Lyapunov-based Q-law guidance for low-thrust orbit transfers.

    Targets the primary Keplerian elements $[a, e, i, \Omega, \omega]$ by
    minimizing a Lyapunov function based on proximity to target elements.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector in ECI frame [m] (3,).
    v_eci : np.ndarray
        Velocity vector in ECI frame [m/s] (3,).
    target_oe : np.ndarray
        Target Keplerian elements $[a, e, i, \Omega, \omega]$ (5,).
        Units should be consistent (m for semi-major axis, rad for angles).
    mu : float
        Gravitational parameter (m^3/s^2).
    accel_max : float
        Maximum thrust acceleration magnitude available (m/s^2).
    weights : np.ndarray, optional
        Weights for each orbital element in the Q function (5,).
        If None, default weights are applied.

    Returns
    -------
    np.ndarray
        Optimal thrust acceleration vector in ECI frame (m/s^2).
    """
    a, e, i, raan, argp, nu, _, _, p, _, _, _ = eci2kepler(r_eci, v_eci)

    oe_curr = np.array([a, e, i, raan, argp])
    oe_target = target_oe[:5]

    if weights is None:
        # Default weighting: normalize a by itself, others keep 1.0
        weights = np.array([1.0 / a**2 if a != 0 else 1.0, 1.0, 1.0, 1.0, 1.0])

    r_mag = np.linalg.norm(r_eci)
    h_vec = np.cross(r_eci, v_eci)
    h_mag = np.linalg.norm(h_vec)
    theta = argp + nu

    # Gauss Variational Equations (GVE) in RTN frame
    # B matrix maps acceleration [radial, tangential, normal] to element rates
    b_a = (2 * a**2 / h_mag) * np.array([e * np.sin(nu), p / r_mag, 0.0])
    b_e = (1 / h_mag) * np.array([p * np.sin(nu), (p + r_mag) * np.cos(nu) + r_mag * e, 0.0])
    b_i = (r_mag * np.cos(theta) / h_mag) * np.array([0.0, 0.0, 1.0])

    if np.sin(i) == 0:
        b_raan = np.zeros(3)
    else:
        b_raan = (r_mag * np.sin(theta) / (h_mag * np.sin(i))) * np.array([0.0, 0.0, 1.0])

    term1 = (1 / (h_mag * e)) * np.array([-p * np.cos(nu), (p + r_mag) * np.sin(nu), 0.0])
    if np.sin(i) == 0:
        term2 = np.zeros(3)
    else:
        term2 = (r_mag * np.sin(theta) * np.cos(i) / (h_mag * np.sin(i))) * np.array(
            [0.0, 0.0, 1.0]
        )
    b_argp = term1 - term2

    b_mat = np.vstack([b_a, b_e, b_i, b_raan, b_argp])

    # Gradient of Lyapunov function Q = sum W_i * (oe_i - target_i)^2
    grad_q_oe = 2 * weights * (oe_curr - oe_target)

    # Optimal thrust direction in RTN is anti-parallel to grad_q_oe @ B
    direction = -(grad_q_oe @ b_mat)

    if np.linalg.norm(direction) < 1e-12:
        return np.zeros(3, dtype=float)

    u_rtn = direction / np.linalg.norm(direction)
    f_rtn = accel_max * u_rtn

    # Convert RTN to ECI
    u_r = r_eci / r_mag
    u_n = h_vec / h_mag
    u_t = np.cross(u_n, u_r)

    r_rtn_to_eci = np.column_stack([u_r, u_t, u_n])
    return np.asarray(r_rtn_to_eci @ f_rtn)


def zem_zev_guidance(
    pos: np.ndarray,
    vel: np.ndarray,
    pos_target: np.ndarray,
    vel_target: np.ndarray,
    t_go: float,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Zero-Effort Miss (ZEM) and Zero-Effort Velocity (ZEV) feedback guidance.

    Commonly used for terminal intercept, rendezvous, or planetary landing.

    Parameters
    ----------
    pos : np.ndarray
        Current position vector (m).
    vel : np.ndarray
        Current velocity vector (m/s).
    pos_target : np.ndarray
        Target position vector (m).
    vel_target : np.ndarray
        Target velocity vector (m/s).
    t_go : float
        Time-to-go until intercept or landing (s).
    gravity : np.ndarray, optional
        Local gravity acceleration vector (m/s^2). Default is zero.

    Returns
    -------
    np.ndarray
        Commanded acceleration vector (m/s^2).
    """
    if t_go <= 1e-6:
        return np.zeros(3, dtype=float)

    if gravity is None:
        gravity = np.zeros(3)

    # Compute ZEM: Miss distance if no further control is applied
    zem = pos_target - (pos + vel * t_go + 0.5 * gravity * t_go**2)
    # Compute ZEV: Velocity error if no further control is applied
    zev = vel_target - (vel + gravity * t_go)

    # Optimal acceleration for minimum effort (integral of a^2)
    return np.asarray((6.0 / t_go**2) * zem - (2.0 / t_go) * zev)


def gravity_turn_guidance(vel: np.ndarray, accel_mag: float, mode: str = "descent") -> np.ndarray:
    """
    Gravity turn steering law.

    Thrust is aligned with the velocity vector to minimize gravity losses and
    pointing errors during ascent or descent.

    Parameters
    ----------
    vel : np.ndarray
        Current velocity vector (m/s).
    accel_mag : float
        Thrust acceleration magnitude to apply (m/s^2).
    mode : str, optional
        'descent' (anti-parallel to vel) or 'ascent' (parallel to vel).

    Returns
    -------
    np.ndarray
        Thrust acceleration vector (m/s^2).
    """
    v_mag = np.linalg.norm(vel)
    if v_mag < 1e-6:
        return np.zeros(3, dtype=float)

    u_v = vel / v_mag

    if mode == "descent":
        return np.asarray(-accel_mag * u_v)
    return np.asarray(accel_mag * u_v)


def apollo_dps_guidance(
    t_go: float,
    pos: np.ndarray,
    vel: np.ndarray,
    pos_t: np.ndarray,
    vel_t: np.ndarray,
    accel_t: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """
    E-guidance (Apollo Descent Propulsion System style).

    A polynomial guidance law targeting specific state and acceleration conditions.

    Parameters
    ----------
    t_go : float
        Time-to-go (s).
    pos : np.ndarray
        Current position (m).
    vel : np.ndarray
        Current velocity (m/s).
    pos_t : np.ndarray
        Target position (m).
    vel_t : np.ndarray
        Target velocity (m/s).
    accel_t : np.ndarray
        Target acceleration (m/s^2).
    gravity : np.ndarray, optional
        Local gravity acceleration (m/s^2).

    Returns
    -------
    np.ndarray
        Commanded acceleration vector (m/s^2).
    """
    if t_go <= 1e-6:
        return np.asarray(accel_t)

    if gravity is None:
        gravity = np.zeros(3)

    delta_r = pos_t - (pos + vel * t_go + 0.5 * gravity * t_go**2)
    delta_v = vel_t - (vel + gravity * t_go)

    return np.asarray(accel_t + (12.0 / t_go**2) * delta_r + (6.0 / t_go) * delta_v)


def indirect_optimal_guidance(
    r0: np.ndarray, v0: np.ndarray, rf: np.ndarray, vf: np.ndarray, tf: float, mu: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Solve minimum-energy transfer using Pontryagin's Minimum Principle (PMP).

    Numerical solution of the indirect optimal control problem formulated as a
    Boundary Value Problem (BVP). The objective is to minimize the integral of
    the square of the acceleration magnitude (control effort).

    Parameters
    ----------
    r0, v0 : np.ndarray
        Initial position and velocity vectors (3,).
    rf, vf : np.ndarray
        Target position and velocity vectors (3,).
    tf : float
        Transfer time (s).
    mu : float
        Gravitational parameter (m^3/s^2).

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        - Time array (N,) if successful, else None.
        - Optimal acceleration profile (3, N) if successful, else None.
    """
    # solve_bvp moved to module level imports

    def odes(t_eval: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        System of ordinary differential equations for states and costates.

        Parameters
        ----------
        t_eval : np.ndarray
            Time points for evaluation.
        state : np.ndarray
            Current state vector [r, v, lambda_r, lambda_v] at t_eval.
            Shape (12, N) where N is number of time points.

        Returns
        -------
        np.ndarray
            Derivatives of the state vector [dr, dv, d(lambda_r), d(lambda_v)].
            Shape (12, N).
        """
        pos = state[:3]
        vel = state[3:6]
        l_r = state[6:9]
        l_v = state[9:12]

        r_mag = np.linalg.norm(pos, axis=0)

        # State dynamics
        dr = vel
        # Optimal control u = -lambda_v for minimizing integral(u^2)
        dv = -(mu / r_mag**3) * pos - l_v

        # Costate dynamics: dot(lambda) = -dH/dx
        # H = 0.5 * u^2 + lambda_r * v + lambda_v * (g + u)
        # dH/dr = lambda_v * dg/dr
        # dH/dv = lambda_r
        # dH/d(lambda_r) = v
        # dH/d(lambda_v) = g + u

        # Gravity gradient term: dg/dr = -(mu/r^3)I + 3(mu/r^5)r*r^T
        # d(lambda_r)/dt = -dH/dr = -lambda_v * dg/dr
        dl_r = np.zeros_like(l_r)
        for idx in range(pos.shape[1]):
            pi = pos[:, idx]
            ri = r_mag[idx]
            lvi = l_v[:, idx]
            grad_g = -(mu / ri**3) * (np.eye(3) - 3 * np.outer(pi, pi) / ri**2)
            dl_r[:, idx] = -grad_g @ lvi

        # d(lambda_v)/dt = -dH/dv = -lambda_r
        dl_v = -l_r

        return np.asarray(np.vstack([dr, dv, dl_r, dl_v]))

    def boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Boundary conditions for the BVP.

        Parameters
        ----------
        ya : np.ndarray
            State vector at initial time t=0.
        yb : np.ndarray
            State vector at final time t=tf.

        Returns
        -------
        np.ndarray
            Array of residuals for the boundary conditions.
        """
        # Initial position and velocity
        # Final position and velocity
        return np.asarray(np.concatenate([ya[:3] - r0, ya[3:6] - v0, yb[:3] - rf, yb[3:6] - vf]))

    t_init = np.linspace(0, tf, 5)
    y_init = np.zeros((12, t_init.size))
    # Linear guess for states
    for i in range(3):
        y_init[i] = np.linspace(r0[i], rf[i], t_init.size)
        y_init[i + 3] = np.linspace(v0[i], vf[i], t_init.size)

    res = solve_bvp(odes, boundary_conditions, t_init, y_init)

    if res.success:
        l_v_sol = res.y[9:12]
        # Optimal acceleration is u = -lambda_v
        return np.asarray(res.x), np.asarray(-l_v_sol)
    return None, None


def direct_collocation_guidance(
    r0: np.ndarray,
    v0: np.ndarray,
    rf: np.ndarray,
    vf: np.ndarray,
    tf: float,
    mu: float,
    n_nodes: int = 20,
) -> np.ndarray | None:
    """
    Solve optimal transfer using Trapezoidal Direct Collocation.

    This method discretizes the trajectory into `n_nodes` and solves for the
    state and control (acceleration) at each node, subject to dynamic and
    boundary constraints. The objective is to minimize the sum of acceleration
    squared (control effort proxy).

    Parameters
    ----------
    r0, v0 : np.ndarray
        Initial state vectors (3,).
    rf, vf : np.ndarray
        Final target state vectors (3,).
    tf : float
        Fixed transfer time (s).
    mu : float
        Gravitational parameter (m^3/s^2).
    n_nodes : int, optional
        Number of discretization nodes. Default is 20.

    Returns
    -------
    Optional[np.ndarray]
        Optimized nodes of shape (n_nodes, 9) containing [r, v, a] at each node.
        Returns None if optimization fails.
    """
    # minimize moved to module level imports

    dt_step = tf / (n_nodes - 1)

    def objective(x_opt: np.ndarray) -> float:
        """
        Objective function to minimize: sum of squared acceleration magnitudes.

        Parameters
        ----------
        x_opt : np.ndarray
            Flattened array of all states and controls [r, v, a] for all nodes.

        Returns
        -------
        float
            Sum of squared acceleration magnitudes.
        """
        # x_opt is flattened: [r1, v1, a1, ..., rn, vn, an]
        # Each node has 9 elements (3 for r, 3 for v, 3 for a)
        accs = x_opt.reshape((n_nodes, 9))[:, 6:9]
        return float(np.sum(np.linalg.norm(accs, axis=1) ** 2))

    def dynamic_constraints(x_opt: np.ndarray) -> np.ndarray:
        """
        Constraints for the optimization problem.

        Includes dynamic constraints (trapezoidal integration) and boundary conditions.

        Parameters
        ----------
        x_opt : np.ndarray
            Flattened array of all states and controls [r, v, a] for all nodes.

        Returns
        -------
        np.ndarray
            Array of residuals for all constraints.
        """
        nodes = x_opt.reshape((n_nodes, 9))
        cons = []

        # Dynamic constraints (Trapezoidal integration)
        for i in range(n_nodes - 1):
            r_i, v_i, a_i = nodes[i, :3], nodes[i, 3:6], nodes[i, 6:9]
            r_next, v_next, a_next = nodes[i + 1, :3], nodes[i + 1, 3:6], nodes[i + 1, 6:9]

            g_i = -(mu / np.linalg.norm(r_i) ** 3) * r_i
            g_next = -(mu / np.linalg.norm(r_next) ** 3) * r_next

            # r_next = r_i + 0.5 * dt * (v_i + v_next)
            cons.append(r_next - r_i - 0.5 * dt_step * (v_i + v_next))
            # v_next = v_i + 0.5 * dt * ( (g_i + a_i) + (g_next + a_next) )
            cons.append(v_next - v_i - 0.5 * dt_step * (g_i + a_i + g_next + a_next))

        # Boundary constraints
        cons.append(nodes[0, :3] - r0)
        cons.append(nodes[0, 3:6] - v0)
        cons.append(nodes[-1, :3] - rf)
        cons.append(nodes[-1, 3:6] - vf)

        return np.asarray(np.concatenate([c.flatten() for c in cons]))

    # Initial guess: linear interpolation for position and velocity, zero acceleration
    x_guess = np.zeros(n_nodes * 9)
    for i in range(n_nodes):
        frac = i / (n_nodes - 1)
        x_guess[i * 9 : i * 9 + 3] = r0 + frac * (rf - r0)
        x_guess[i * 9 + 3 : i * 9 + 6] = v0 + frac * (vf - v0)
        # Accelerations are initialized to zero

    res = minimize(
        objective, x_guess, constraints={"type": "eq", "fun": dynamic_constraints}, method="SLSQP"
    )

    if res.success:
        return np.asarray(res.x.reshape((n_nodes, 9)))
    return None
