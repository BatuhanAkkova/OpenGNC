"""
End-to-End Closed-Loop Spacecraft Mission Simulation with Autonomous FDIR Fault Injections.

Simulates closed-loop 3-axis attitude tracking under:
1. Nominal operation (0 - 50s).
2. Star Tracker blinding / step bias injection (50s) -> Chi-Square detection & isolation.
3. Reaction Wheel 1 motor failure (100s) -> 4-to-3 RW dynamic reallocation.
4. Reaction Wheel 2 motor failure (150s) -> Autonomous transition to Underactuated / Safe Mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengnc.fdir.actuator_fdir import (
    ActuatorFaultDetector,
    ActuatorHealth,
    WeightedControlAllocator,
)
from opengnc.fdir.executive import (
    FDIRMode,
    FDIRExecutive,
)
from opengnc.fdir.sensor_fdir import (
    ChiSquareMonitor,
    SensorHealth,
    SensorIsolationManager,
)


def test_fdir_e2e_mission_scenario():
    """Execute 200s closed-loop simulation with cascading sensor and actuator fault injections."""
    dt = 0.1  # 10 Hz flight loop
    total_time = 200.0
    steps = int(total_time / dt)

    # Spacecraft Inertia
    J = np.diag([15.0, 15.0, 20.0])
    J_inv = np.linalg.inv(J)

    # 4-wheel pyramid configuration at 30 deg inclination
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    B = np.array(
        [
            [c, 0.0, -c, 0.0],
            [0.0, c, 0.0, -c],
            [s, s, s, s],
        ]
    )

    allocator = WeightedControlAllocator(B, u_min=-0.2, u_max=0.2)
    sensor_mgr = SensorIsolationManager(["ST1", "ST2"], max_consecutive_faults=3)
    st1_monitor = ChiSquareMonitor(dim=3, confidence=0.99, window_size=5, persistence_threshold=3)
    st2_monitor = ChiSquareMonitor(dim=3, confidence=0.99, window_size=5, persistence_threshold=3)
    act_detector = ActuatorFaultDetector(
        max_wheel_speed=600.0, torque_tolerance=0.01, persistence_steps=3
    )
    for i in range(4):
        act_detector.register_actuator(f"RW{i}")

    executive = FDIRExecutive(initial_mode=FDIRMode.NOMINAL, max_rate_runaway=0.2)

    # Initial state
    q = np.array([0.05, -0.04, 0.02, 0.998])
    q /= np.linalg.norm(q)
    w = np.array([0.005, -0.005, 0.002])  # rad/s

    # Target: origin
    q_target = np.array([0.0, 0.0, 0.0, 1.0])

    # Controller gains
    kp = 2.0
    kd = 5.0

    mode_at_40s = None
    mode_at_80s = None
    mode_at_120s = None
    mode_at_180s = None

    pointing_error_at_80s = None
    pointing_error_at_120s = None

    for step in range(steps):
        t = step * dt

        # 1. Sensor Measurements & Fault Injections
        # True attitude vector part
        z_true = q[:3]

        # Sensor 1: Blinding / huge bias fault starting at t >= 50s
        z_st1 = z_true.copy() + np.random.normal(0, 0.001, 3)
        if t >= 50.0:
            z_st1 += np.array([0.1, -0.1, 0.05])  # Injected sensor blinding

        # Sensor 2: Stays healthy
        z_st2 = z_true.copy() + np.random.normal(0, 0.001, 3)

        # Innovation monitoring against estimated state (for test, against nominal model)
        S_cov = np.eye(3) * 0.0001
        res_st1 = z_st1 - z_true
        res_st2 = z_st2 - z_true

        f_st1, _, _ = st1_monitor.update(res_st1, S_cov)
        f_st2, _, _ = st2_monitor.update(res_st2, S_cov)

        st1_health = sensor_mgr.report_status("ST1", has_fault=f_st1)
        st2_health = sensor_mgr.report_status("ST2", has_fault=f_st2)

        # Sensor fusion: if ST1 isolated, rely on ST2
        if st1_health == SensorHealth.ISOLATED:
            q_est_vec = z_st2
        elif st2_health == SensorHealth.ISOLATED:
            q_est_vec = z_st1
        else:
            q_est_vec = 0.5 * (z_st1 + z_st2)

        # 2. Attitude Control Law
        tau_cmd = -kp * q_est_vec - kd * w

        # 3. Control Allocation
        # Actuator fault injections:
        # At t >= 100s: RW0 motor lockup (zero torque response)
        # At t >= 150s: RW1 motor lockup (zero torque response)
        try:
            u_cmd = allocator.allocate(tau_cmd)
        except RuntimeError:
            # Underactuated condition reached
            u_cmd = np.zeros(4)

        u_actual = u_cmd.copy()
        if t >= 100.0:
            u_actual[0] = 0.0  # Wheel 0 dead
        if t >= 150.0:
            u_actual[1] = 0.0  # Wheel 1 dead

        # Check actuator health telemetry with tachometer overspeed runaway on failed wheels
        speeds = [
            750.0 if t >= 100.0 else 100.0,
            750.0 if t >= 150.0 else 100.0,
            100.0,
            100.0,
        ]
        for i in range(4):
            rw_status = act_detector.check_wheel(
                f"RW{i}", u_cmd[i], u_actual[i], current_speed=speeds[i]
            )
            allocator.set_actuator_health(i, rw_status)

        # 4. FDIR Executive Mode Evaluation
        pointing_err_deg = np.degrees(2.0 * np.linalg.norm(q[:3]))
        current_mode = executive.evaluate_state(
            time_s=t,
            sensor_health=sensor_mgr.sensors,
            actuator_health=act_detector.actuator_status,
            pointing_error_deg=pointing_err_deg,
            rate_norm=float(np.linalg.norm(w)),
        )

        # Sample modes at checkpoints
        if abs(t - 40.0) < dt / 2:
            mode_at_40s = current_mode
        elif abs(t - 80.0) < dt / 2:
            mode_at_80s = current_mode
            pointing_error_at_80s = pointing_err_deg
        elif abs(t - 120.0) < dt / 2:
            mode_at_120s = current_mode
            pointing_error_at_120s = pointing_err_deg
        elif abs(t - 180.0) < dt / 2:
            mode_at_180s = current_mode

        # 5. Physics Dynamics Propagation
        # Angular acceleration: w_dot = J^-1 * (tau_total - w x (J * w))
        tau_total = B @ u_actual
        gyroscopic = np.cross(w, J @ w)
        w_dot = J_inv @ (tau_total - gyroscopic)
        w += w_dot * dt

        # Quaternion kinematics: q_dot = 0.5 * Omega(w) * q
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-12:
            dq = np.array(
                [
                    w[0] / w_norm * np.sin(w_norm * dt / 2),
                    w[1] / w_norm * np.sin(w_norm * dt / 2),
                    w[2] / w_norm * np.sin(w_norm * dt / 2),
                    np.cos(w_norm * dt / 2),
                ]
            )
            # Quaternion multiplication
            q_new = np.array(
                [
                    q[3] * dq[0] + q[0] * dq[3] + q[1] * dq[2] - q[2] * dq[1],
                    q[3] * dq[1] - q[0] * dq[2] + q[1] * dq[3] + q[2] * dq[0],
                    q[3] * dq[2] + q[0] * dq[1] - q[1] * dq[0] + q[2] * dq[3],
                    q[3] * dq[3] - q[0] * dq[0] - q[1] * dq[1] - q[2] * dq[2],
                ]
            )
            q = q_new / np.linalg.norm(q_new)

    # ASSERTIONS: End-to-end mission scenario verification
    # 1. At 40s (pre-fault): nominal operation
    assert mode_at_40s == FDIRMode.NOMINAL

    # 2. At 80s (post ST1 failure): ST1 isolated, ST2 maintains pointing under 1 deg
    assert mode_at_80s == FDIRMode.DEGRADED_POINTING
    assert sensor_mgr.sensors["ST1"] == SensorHealth.ISOLATED
    assert pointing_error_at_80s is not None
    assert pointing_error_at_80s < 1.0  # Pointing kept tight via ST2!

    # 3. At 120s (post RW0 failure): 4-to-3 reallocation maintains pointing
    assert mode_at_120s == FDIRMode.DEGRADED_POINTING
    assert act_detector.actuator_status["RW0"] == ActuatorHealth.FAILED
    assert pointing_error_at_120s is not None
    assert pointing_error_at_120s < 1.0  # Pointing kept tight via 3 remaining wheels!

    # 4. At 180s (post RW1 failure): 2 wheels remain -> autonomous transition to UNDERACTUATED_CONTROL
    assert mode_at_180s == FDIRMode.UNDERACTUATED_CONTROL
    assert len(executive.transition_log) >= 2
