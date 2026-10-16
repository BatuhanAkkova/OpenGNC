"""
Actuator Fault Detection, Isolation, and Reconfiguration (FTC).

Provides health monitoring for reaction wheels and thrusters, dynamic weighted
pseudo-inverse control allocation with health masking, and underactuated
attitude control for 2-wheel degraded recovery.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.optimize import minimize


class ActuatorHealth(Enum):
    """Health classification for spacecraft actuators."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ActuatorFaultDetector:
    """
    Monitors reaction wheel and thruster health states.

    Detects motor lockup, friction torque spikes, tachometer runaway,
    and command-to-telemetry tracking error.

    Parameters
    ----------
    max_wheel_speed : float, optional
        Maximum allowable reaction wheel speed (rad/s). Default is 600.0 (~5700 RPM).
    torque_tolerance : float, optional
        Allowable torque discrepancy between command and telemetry (N*m).
        Default is 0.02.
    persistence_steps : int, optional
        Number of consecutive timesteps with violation before marking FAILED.
        Default is 5.
    """

    def __init__(
        self,
        max_wheel_speed: float = 600.0,
        torque_tolerance: float = 0.02,
        persistence_steps: int = 5,
    ) -> None:
        self.max_speed = float(max_wheel_speed)
        self.torque_tol = float(torque_tolerance)
        self.persistence = persistence_steps
        self.violation_counts: dict[str, int] = {}
        self.actuator_status: dict[str, ActuatorHealth] = {}

    def register_actuator(self, name: str) -> None:
        """Register a new actuator to monitor."""
        self.violation_counts[name] = 0
        self.actuator_status[name] = ActuatorHealth.HEALTHY

    def check_wheel(
        self,
        name: str,
        cmd_torque: float,
        actual_torque: float,
        current_speed: float,
    ) -> ActuatorHealth:
        """
        Evaluate wheel telemetry against health thresholds.

        Parameters
        ----------
        name : str
            Actuator identifier.
        cmd_torque : float
            Commanded torque (N*m).
        actual_torque : float
            Measured/estimated torque output (N*m).
        current_speed : float
            Current wheel spin rate (rad/s).

        Returns
        -------
        ActuatorHealth
            Updated health status.
        """
        if name not in self.actuator_status:
            self.register_actuator(name)

        if self.actuator_status[name] == ActuatorHealth.FAILED:
            return ActuatorHealth.FAILED

        # Check conditions
        speed_overshoot = abs(current_speed) >= self.max_speed
        torque_mismatch = abs(cmd_torque - actual_torque) > self.torque_tol

        if speed_overshoot or torque_mismatch:
            self.violation_counts[name] += 1
            if self.violation_counts[name] >= self.persistence:
                self.actuator_status[name] = ActuatorHealth.FAILED
            else:
                self.actuator_status[name] = ActuatorHealth.DEGRADED
        else:
            self.violation_counts[name] = max(0, self.violation_counts[name] - 1)
            if self.violation_counts[name] == 0:
                self.actuator_status[name] = ActuatorHealth.HEALTHY

        return self.actuator_status[name]

    def set_failed(self, name: str) -> None:
        """Manually mark an actuator as FAILED."""
        if name in self.actuator_status:
            self.actuator_status[name] = ActuatorHealth.FAILED
            self.violation_counts[name] = self.persistence


class WeightedControlAllocator:
    r"""
    Dynamic Weighted Control Allocator (WQP / Weighted Pseudo-Inverse).

    Maps commanded 3-axis control torque $\boldsymbol{\tau}_c \in \mathbb{R}^3$
    to actuator commands $\mathbf{u} \in \mathbb{R}^n$ via:
    $$\min_{\mathbf{u}} \frac{1}{2} \mathbf{u}^T \mathbf{W}_u \mathbf{u} \quad \text{s.t.} \quad \mathbf{B} \mathbf{u} = \boldsymbol{\tau}_c$$
    subject to $\mathbf{u}_{\min} \le \mathbf{u} \le \mathbf{u}_{\max}$.

    When an actuator fails, its weight $W_{u, ii}$ is scaled to $\infty$ (or
    its column in $\mathbf{B}$ is masked to zero), redistributing torque
    instantly to remaining functional actuators without perturbing the attitude loop.

    Parameters
    ----------
    b_matrix : np.ndarray
        Actuator geometry/effectiveness matrix $\mathbf{B} \in \mathbb{R}^{3 \times n}$.
    u_min : np.ndarray | float, optional
        Minimum torque limit per actuator (N*m). Default is -0.1 N*m.
    u_max : np.ndarray | float, optional
        Maximum torque limit per actuator (N*m). Default is +0.1 N*m.
    """

    def __init__(
        self,
        b_matrix: np.ndarray,
        u_min: np.ndarray | float = -0.1,
        u_max: np.ndarray | float = 0.1,
    ) -> None:
        self.B = np.asarray(b_matrix, dtype=float)
        if self.B.shape[0] != 3:
            raise ValueError(f"B matrix must have 3 rows for 3-axis torque, got {self.B.shape}")

        self.num_actuators = self.B.shape[1]
        if np.isscalar(u_min):
            val_min = float(np.asarray(u_min).item())
            self.u_min: np.ndarray = np.full(self.num_actuators, val_min, dtype=float)
        else:
            self.u_min = np.asarray(u_min, dtype=float)

        if np.isscalar(u_max):
            val_max = float(np.asarray(u_max).item())
            self.u_max: np.ndarray = np.full(self.num_actuators, val_max, dtype=float)
        else:
            self.u_max = np.asarray(u_max, dtype=float)

        self.weights = np.ones(self.num_actuators, dtype=float)
        self.active_mask = np.ones(self.num_actuators, dtype=bool)

    def set_actuator_health(self, index: int, health: ActuatorHealth) -> None:
        r"""
        Update weighting and active masking based on actuator health.

        Parameters
        ----------
        index : int
            Actuator channel index ($0 \le i < n$).
        health : ActuatorHealth
            Health state from FDI monitor.
        """
        if not (0 <= index < self.num_actuators):
            raise IndexError(f"Actuator index {index} out of range (num={self.num_actuators})")

        if health == ActuatorHealth.HEALTHY:
            self.weights[index] = 1.0
            self.active_mask[index] = True
        elif health == ActuatorHealth.DEGRADED:
            self.weights[index] = 100.0  # Penalize high usage
            self.active_mask[index] = True
        else:  # FAILED
            self.weights[index] = 1e8
            self.active_mask[index] = False

    def allocate(self, tau_cmd: np.ndarray) -> np.ndarray:
        r"""
        Allocate commanded 3-axis torque to individual actuators.

        Parameters
        ----------
        tau_cmd : np.ndarray
            Commanded torque vector $\boldsymbol{\tau}_c \in \mathbb{R}^3$ (N*m).

        Returns
        -------
        np.ndarray
            Actuator torque vector $\mathbf{u} \in \mathbb{R}^n$ (N*m).
        """
        tau = np.asarray(tau_cmd, dtype=float).flatten()
        if len(tau) != 3:
            raise ValueError(f"tau_cmd must be a 3-element vector, got length {len(tau)}")

        active_indices = np.where(self.active_mask)[0]
        num_active = len(active_indices)

        if num_active < 3:
            raise RuntimeError(
                f"Underactuated condition: only {num_active} actuators active. "
                "Switch to UnderactuatedAttitudeController."
            )

        # Sub-matrix for active actuators
        B_sub = self.B[:, active_indices]
        W_sub_inv = np.diag(1.0 / self.weights[active_indices])

        # Analytical Weighted Pseudo-Inverse:
        # u = W^-1 * B^T * (B * W^-1 * B^T)^-1 * tau
        BW = B_sub @ W_sub_inv
        BWB_t = BW @ B_sub.T

        try:
            inv_BWB_t = np.linalg.inv(BWB_t)
            u_active = W_sub_inv @ B_sub.T @ (inv_BWB_t @ tau)
        except np.linalg.LinAlgError:
            # Degenerate geometry fallback: Moore-Penrose pinv
            u_active = np.linalg.pinv(B_sub) @ tau

        # Check if saturated
        is_saturated = np.any(u_active < self.u_min[active_indices]) or np.any(
            u_active > self.u_max[active_indices]
        )

        if is_saturated:
            # Constrained Quadratic Program via SLSQP
            def obj(u_vec: np.ndarray) -> float:
                return 0.5 * float(u_vec.T @ np.diag(self.weights[active_indices]) @ u_vec)

            def eq_cons(u_vec: np.ndarray) -> np.ndarray:
                return np.asarray(B_sub @ u_vec - tau, dtype=float)

            bounds = [(self.u_min[idx], self.u_max[idx]) for idx in active_indices]
            res = minimize(
                obj,
                x0=np.clip(u_active, self.u_min[active_indices], self.u_max[active_indices]),
                bounds=bounds,
                constraints={"type": "eq", "fun": eq_cons},
                method="SLSQP",
            )
            if res.success:
                u_active = np.asarray(res.x, dtype=float)
            else:
                # If exact equality cannot be met under bounds, clip
                u_active = np.clip(u_active, self.u_min[active_indices], self.u_max[active_indices])

        # Reconstruct full actuator command vector
        u_full = np.zeros(self.num_actuators, dtype=float)
        u_full[active_indices] = u_active
        return u_full


class UnderactuatedAttitudeController:
    """
    Underactuated Spacecraft Attitude Controller (2-Reaction-Wheel Recovery).

    When one or two reaction wheels fail, leaving only 2 functional reaction
    wheels, full 3-axis instantaneous controllability via wheels alone is lost.
    This controller stabilizes the two actuated body axes using high-gain PD
    control and uses auxiliary magnetic torquers (or cross-coupling precession)
    to asymptotically damp momentum along the unactuated body axis.

    Parameters
    ----------
    actuated_axes : tuple[int, int]
        Indices of the two functional body axes (e.g. (0, 1) for X and Y).
    kp : float, optional
        Proportional attitude error gain. Default is 0.5.
    kd : float, optional
        Derivative angular rate damping gain. Default is 2.0.
    km : float, optional
        Magnetic dipole cross-product gain for unactuated axis. Default is 1000.0.
    """

    def __init__(
        self,
        actuated_axes: tuple[int, int] = (0, 1),
        kp: float = 0.5,
        kd: float = 2.0,
        km: float = 1000.0,
    ) -> None:
        self.actuated_axes = actuated_axes
        all_axes = {0, 1, 2}
        unactuated = all_axes - set(actuated_axes)
        if len(unactuated) != 1:
            raise ValueError(f"Must have exactly 2 actuated axes, got {actuated_axes}")
        self.unactuated_axis = list(unactuated)[0]

        self.kp = float(kp)
        self.kd = float(kd)
        self.km = float(km)

    def compute_control(
        self,
        q_err: np.ndarray,
        w_err: np.ndarray,
        b_field_body: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute control torques for the 2 functional wheels and auxiliary magnetic dipole.

        Parameters
        ----------
        q_err : np.ndarray
            Attitude quaternion error vector $[q_1, q_2, q_3, q_4]^T$ (vector part first 3).
        w_err : np.ndarray
            Body angular velocity error $[w_x, w_y, w_z]^T$ (rad/s).
        b_field_body : np.ndarray, optional
            Local geomagnetic field vector in body frame (Tesla).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (tau_wheels, magnetic_dipole)
            - tau_wheels: 3-element torque vector with zero along the unactuated axis.
            - magnetic_dipole: 3-element commanded magnetic dipole moment $\mathbf{m}$ ($A \cdot m^2$).
        """
        q = np.asarray(q_err, dtype=float).flatten()
        w = np.asarray(w_err, dtype=float).flatten()

        tau_wheels = np.zeros(3, dtype=float)

        # 1. PD control on the two actuated axes
        for axis in self.actuated_axes:
            tau_wheels[axis] = -self.kp * q[axis] - self.kd * w[axis]

        # 2. Auxiliary magnetic torquer desaturation / damping for unactuated axis
        magnetic_dipole = np.zeros(3, dtype=float)
        if b_field_body is not None:
            b = np.asarray(b_field_body, dtype=float).flatten()
            b_norm_sq = float(np.dot(b, b))
            if b_norm_sq > 1e-18:
                # Desired torque along unactuated axis
                tau_des_unact = np.zeros(3, dtype=float)
                tau_des_unact[self.unactuated_axis] = (
                    -self.kp * q[self.unactuated_axis] - self.kd * w[self.unactuated_axis]
                )

                # Command dipole: m = (tau_des x B) / ||B||^2
                magnetic_dipole = np.cross(tau_des_unact, b) / b_norm_sq
                magnetic_dipole = np.clip(magnetic_dipole, -50.0, 50.0)

        return tau_wheels, magnetic_dipole
