"""
Nonlinear Model Predictive Control (NMPC) using CasADi with Multiple Shooting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

    class ca:  # type: ignore
        """Placeholder for CasADi types when not installed."""
        MX = Any
        SX = Any
        def __getattr__(self, name: str) -> Any:
            return Any


class CasadiNMPC:
    """
    High-performance Nonlinear Model Predictive Controller (NMPC) using CasADi.

    Optimizes a finite-horizon cost function subject to nonlinear dynamics and
    algebraic constraints using a Multiple Shooting formulation for numerical
    stability and parallelism.

    NOTE: Requires the 'casadi' package (`pip install opengnc[mpc]`).

    Parameters
    ----------
    nx : int
        State dimension.
    nu : int
        Input dimension.
    horizon : int
        Prediction horizon N.
    dt : float
        Time step (s).
    dynamics_func : Callable
        System dynamics f(x, u). Returns x_next if discrete, else dx/dt.
    cost_func : Callable
        Stage cost function L(x, u).
    terminal_cost_func : Callable
        Terminal cost function V(x).
    u_min, u_max : float or np.ndarray, optional
        Control input constraints.
    x_min, x_max : float or np.ndarray, optional
        State trajectory constraints.
    discrete : bool, optional
        If True, the dynamics function is assumed discrete. If False, RK4
        integration is performed internally. Default is False.
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        horizon: int,
        dt: float,
        dynamics_func: Callable,
        cost_func: Callable,
        terminal_cost_func: Callable,
        u_min: float | np.ndarray | None = None,
        u_max: float | np.ndarray | None = None,
        x_min: float | np.ndarray | None = None,
        x_max: float | np.ndarray | None = None,
        discrete: bool = False,
    ) -> None:
        """Initialize and formulate the CasADi NLP problem."""
        if not CASADI_AVAILABLE:
            raise ImportError(
                "CasADi is not installed. Please install with 'pip install opengnc[mpc]' "
                "to use the CasadiNMPC controller."
            )

        self.nx = int(nx)
        self.nu = int(nu)
        self.N = int(horizon)
        self.dt = float(dt)
        self.f = dynamics_func
        self.L = cost_func
        self.V = terminal_cost_func
        self.discrete = discrete

        self.u_min = self._setup_bounds(u_min, self.nu, -np.inf)
        self.u_max = self._setup_bounds(u_max, self.nu, np.inf)
        self.x_min = self._setup_bounds(x_min, self.nx, -np.inf)
        self.x_max = self._setup_bounds(x_max, self.nx, np.inf)
        self.lbx: np.ndarray
        self.ubx: np.ndarray
        self.lbg: np.ndarray
        self.ubg: np.ndarray

        self._setup_solver()

    def _setup_bounds(self, bound: float | np.ndarray | None, dim: int, default_val: float) -> np.ndarray:
        """Helper to convert scalar or array bounds to full-dimension arrays."""
        if bound is None:
            return np.full(dim, default_val)
        bound_arr = np.array(bound).flatten()
        if len(bound_arr) == 1:
            return np.full(dim, bound_arr[0])
        if len(bound_arr) != dim:
            raise ValueError(f"Bound dimension mismatch. Expected {dim}, got {len(bound_arr)}")
        return bound_arr

    def _rk4_step(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Perform a 4th-order Runge-Kutta integration step."""
        k1 = self.f(x, u)
        k2 = self.f(x + self.dt / 2.0 * k1, u)
        k3 = self.f(x + self.dt / 2.0 * k2, u)
        k4 = self.f(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _setup_solver(self) -> None:
        """Formulate the NLP and instantiate the IPOPT solver."""
        # Symbolic decision variables
        X = ca.MX.sym("X", self.nx, self.N + 1)  # State variables
        U = ca.MX.sym("U", self.nu, self.N)       # Control variables
        X0 = ca.MX.sym("X0", self.nx)             # Initial state parameter

        cost = 0.0
        g = []  # Constraints (Multiple Shooting)
        g.append(X[:, 0] - X0)  # Initial condition constraint

        for k in range(self.N):
            cost += self.L(X[:, k], U[:, k])

            x_next_sim = self._rk4_step(X[:, k], U[:, k]) if not self.discrete else self.f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next_sim)

        cost += self.V(X[:, self.N])

        # Formulate decision vector and symbolic NLP
        v_decision = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g_constraints = ca.vertcat(*g)
        nlp = {"x": v_decision, "f": cost, "g": g_constraints, "p": X0}

        # Solver configuration (IPOPT)
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": 1e-6,
            "ipopt.max_iter": 150
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Build concatenated bound vectors for X and U
        lbx_list: list[float] = []
        ubx_list: list[float] = []
        for _ in range(self.N + 1):
            lbx_list.extend(self.x_min)
            ubx_list.extend(self.x_max)
        for _ in range(self.N):
            lbx_list.extend(self.u_min)
            ubx_list.extend(self.u_max)

        self.lbx = np.array(lbx_list, dtype=float)
        self.ubx = np.array(ubx_list, dtype=float)
        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))

    def solve(self, x0: np.ndarray, u_guess: np.ndarray | None = None) -> np.ndarray:
        """
        Solve the NMPC problem for the given initial state.

        Parameters
        ----------
        x0 : np.ndarray
            Current system state (nx,).
        u_guess : np.ndarray, optional
            Initial guess for control trajectory (N, nu).

        Returns
        -------
        np.ndarray
            Optimal control trajectory (N, nu).
        """
        x_init = np.asarray(x0).flatten()

        # Warm start guess
        x_start = np.tile(x_init, (self.N + 1, 1)).flatten()
        if u_guess is not None:
            u_start = np.asarray(u_guess).flatten()
            if u_start.size != self.N * self.nu:
                # Resize to match solver expectation if possible, or repeat
                u_start = np.resize(u_start, self.N * self.nu)
        else:
            u_start = np.zeros(self.N * self.nu)
        v_start = np.concatenate([x_start, u_start])

        sol = self.solver(
            x0=v_start, p=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
        )

        v_opt = sol["x"].full().flatten()
        # Extract controls (located after all state variables in decision vector)
        u_opt_flat = v_opt[self.nx * (self.N + 1):]
        return cast(np.ndarray, u_opt_flat.reshape((self.N, self.nu)))
