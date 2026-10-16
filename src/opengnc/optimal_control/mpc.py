"""
Linear and Nonlinear Model Predictive Control (MPC) solvers.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize


class LinearMPC:
    r"""
    Linear Model Predictive Controller (MPC) using SLSQP optimization.

    Solves a finite-horizon optimal control problem for linear discrete-time
    systems subject to state and input constraints.

    Dynamics: $x[k+1] = A x[k] + B u[k]$
    Objective: $\min \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N$

    Parameters
    ----------
    A : np.ndarray
        State transition matrix (nx x nx).
    B : np.ndarray
        Input matrix (nx x nu).
    Q : np.ndarray
        State cost matrix (nx x nx).
    R : np.ndarray
        Input cost matrix (nu x nu).
    horizon : int
        Prediction and control horizon N.
    P : np.ndarray, optional
        Terminal state cost matrix. Defaults to Q.
    u_min, u_max : float or np.ndarray, optional
        Minimum and maximum control input bounds.
    x_min, x_max : float or np.ndarray, optional
        Minimum and maximum state constraints.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        P: np.ndarray | None = None,
        u_min: float | np.ndarray | None = None,
        u_max: float | np.ndarray | None = None,
        x_min: float | np.ndarray | None = None,
        x_max: float | np.ndarray | None = None,
    ) -> None:
        """Initialize the Linear MPC solver."""
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.N = int(horizon)
        self.P = np.asarray(P) if P is not None else self.Q

        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max

        # Dimension checks for bounds
        if self.u_min is not None and not np.isscalar(self.u_min):
            if np.asarray(self.u_min).size != self.nu:
                raise ValueError(f"u_min dimension mismatch. Expected {self.nu}")
        if self.u_max is not None and not np.isscalar(self.u_max):
            if np.asarray(self.u_max).size != self.nu:
                raise ValueError(f"u_max dimension mismatch. Expected {self.nu}")
        if self.x_min is not None and not np.isscalar(self.x_min):
            if np.asarray(self.x_min).size != self.nx:
                raise ValueError(f"x_min dimension mismatch. Expected {self.nx}")
        if self.x_max is not None and not np.isscalar(self.x_max):
            if np.asarray(self.x_max).size != self.nx:
                raise ValueError(f"x_max dimension mismatch. Expected {self.nx}")

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """
        Solve the MPC optimization problem for a given initial state.

        Parameters
        ----------
        x0 : np.ndarray
            Current state of the system (nx,).

        Returns
        -------
        np.ndarray
            Optimal control sequence [u_0, u_1, ..., u_{N-1}] of shape (N, nu).
        """
        x_init = np.asarray(x0).flatten()

        def objective(u_flat: np.ndarray) -> float:
            u_seq = u_flat.reshape((self.N, self.nu))
            cost = 0.0
            x = x_init.copy()

            for k in range(self.N):
                u = u_seq[k]
                cost += x.T @ self.Q @ x + u.T @ self.R @ u
                x = self.A @ x + self.B @ u

            return float(cost + x.T @ self.P @ x)

        # Build Input Bounds
        bounds: list[tuple[float, float]] = []
        if self.u_min is not None or self.u_max is not None:
            umin = (
                np.full(self.nu, self.u_min)
                if np.isscalar(self.u_min)
                else np.asarray(self.u_min).flatten()
            )
            umax = (
                np.full(self.nu, self.u_max)
                if np.isscalar(self.u_max)
                else np.asarray(self.u_max).flatten()
            )
            for _ in range(self.N):
                for i in range(self.nu):
                    bounds.append((umin[i], umax[i]))
        else:
            bounds = []

        # State constraints
        constraints: list[dict[str, Any]] = []
        if self.x_min is not None or self.x_max is not None:
            xmin = (
                np.full(self.nx, self.x_min)
                if np.isscalar(self.x_min)
                else np.asarray(self.x_min).flatten()
            )
            xmax = (
                np.full(self.nx, self.x_max)
                if np.isscalar(self.x_max)
                else np.asarray(self.x_max).flatten()
            )

            def state_constraint_fun(u_flat: np.ndarray) -> np.ndarray:
                u_seq = u_flat.reshape((self.N, self.nu))
                x = x_init.copy()
                residuals = []
                for k in range(self.N):
                    x = self.A @ x + self.B @ u_seq[k]
                    if self.x_min is not None:
                        residuals.extend(x - xmin)
                    if self.x_max is not None:
                        residuals.extend(xmax - x)
                return np.array(residuals)

            constraints.append({"type": "ineq", "fun": state_constraint_fun})

        u_guess = np.zeros(self.N * self.nu)
        res = minimize(
            objective, u_guess, method="SLSQP", bounds=bounds or None, constraints=constraints
        )

        if not res.success:
            # Note: In production GNC, fallback controllers are usually triggered here
            pass

        return cast(np.ndarray, res.x.reshape((self.N, self.nu)))


class NonlinearMPC:
    r"""
    Nonlinear Model Predictive Controller (NMPC).

    Solves a finite-horizon optimal control problem for systems with nonlinear
    dynamics using single-shooting numerical optimization.

    Dynamics: $x[k+1] = f(x[k], u[k])$
    Objective: $\min \sum_{k=0}^{N-1} L(x_k, u_k) + V(x_N)$

    Parameters
    ----------
    dynamics_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Nonlinear transition function $f(x, u)$.
    cost_func : Callable[[np.ndarray, np.ndarray], float]
        Stage cost function $L(x, u)$.
    terminal_cost_func : Callable[[np.ndarray], float]
        Terminal cost function $V(x)$.
    horizon : int
        Prediction horizon N.
    nx : int
        State dimension.
    nu : int
        Input dimension.
    u_min, u_max : float or np.ndarray, optional
        Control input constraints.
    x_min, x_max : float or np.ndarray, optional
        State constraints.
    """

    def __init__(
        self,
        dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cost_func: Callable[[np.ndarray, np.ndarray], float],
        terminal_cost_func: Callable[[np.ndarray], float],
        horizon: int,
        nx: int,
        nu: int,
        u_min: float | np.ndarray | None = None,
        u_max: float | np.ndarray | None = None,
        x_min: float | np.ndarray | None = None,
        x_max: float | np.ndarray | None = None,
    ) -> None:
        """Initialize the Nonlinear MPC solver."""
        self.f = dynamics_func
        self.L = cost_func
        self.V = terminal_cost_func
        self.N = int(horizon)
        self.nx = int(nx)
        self.nu = int(nu)

        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max

        # Dimension checks for bounds
        if self.u_min is not None and not np.isscalar(self.u_min):
            if np.asarray(self.u_min).size != self.nu:
                raise ValueError(f"u_min dimension mismatch. Expected {self.nu}")
        if self.u_max is not None and not np.isscalar(self.u_max):
            if np.asarray(self.u_max).size != self.nu:
                raise ValueError(f"u_max dimension mismatch. Expected {self.nu}")
        if self.x_min is not None and not np.isscalar(self.x_min):
            if np.asarray(self.x_min).size != self.nx:
                raise ValueError(f"x_min dimension mismatch. Expected {self.nx}")
        if self.x_max is not None and not np.isscalar(self.x_max):
            if np.asarray(self.x_max).size != self.nx:
                raise ValueError(f"x_max dimension mismatch. Expected {self.nx}")

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """
        Solve the NMPC optimization problem using single shooting.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state of the system (nx,).

        Returns
        -------
        np.ndarray
            Optimal control sequence (N, nu).
        """
        x_init = np.asarray(x0).flatten()

        def objective(u_flat: np.ndarray) -> float:
            u_seq = u_flat.reshape((self.N, self.nu))
            total_cost = 0.0
            x = x_init.copy()

            for k in range(self.N):
                u = u_seq[k]
                total_cost += self.L(x, u)
                x = np.asarray(self.f(x, u)).flatten()

            return float(total_cost + self.V(x))

        bounds: list[tuple[float, float]] = []
        if self.u_min is not None or self.u_max is not None:
            umin = (
                np.full(self.nu, self.u_min)
                if np.isscalar(self.u_min)
                else np.asarray(self.u_min).flatten()
            )
            umax = (
                np.full(self.nu, self.u_max)
                if np.isscalar(self.u_max)
                else np.asarray(self.u_max).flatten()
            )
            for _ in range(self.N):
                for i in range(self.nu):
                    bounds.append((umin[i], umax[i]))
        else:
            bounds = []

        # State constraints
        constraints: list[dict[str, Any]] = []
        if self.x_min is not None or self.x_max is not None:
            xmin = (
                np.full(self.nx, self.x_min)
                if np.isscalar(self.x_min)
                else np.asarray(self.x_min).flatten()
            )
            xmax = (
                np.full(self.nx, self.x_max)
                if np.isscalar(self.x_max)
                else np.asarray(self.x_max).flatten()
            )

            def state_constraint_fun(u_flat: np.ndarray) -> np.ndarray:
                u_seq = u_flat.reshape((self.N, self.nu))
                x = x_init.copy()
                residuals = []
                for k in range(self.N):
                    x = np.asarray(self.f(x, u_seq[k])).flatten()
                    if self.x_min is not None:
                        residuals.extend(x - xmin)
                    if self.x_max is not None:
                        residuals.extend(xmax - x)
                return np.array(residuals)

            constraints.append({"type": "ineq", "fun": state_constraint_fun})

        u0 = np.zeros(self.N * self.nu)
        res = minimize(
            objective, u0, method="SLSQP", bounds=bounds or None, constraints=constraints
        )

        return cast(np.ndarray, res.x.reshape((self.N, self.nu)))
