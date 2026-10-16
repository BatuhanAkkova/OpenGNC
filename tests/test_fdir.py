"""
Unit tests for the OpenGNC Fault Detection, Isolation, and Recovery (FDIR) Subsystem.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengnc.fdir.actuator_fdir import (
    ActuatorFaultDetector,
    ActuatorHealth,
    UnderactuatedAttitudeController,
    WeightedControlAllocator,
)
from opengnc.fdir.executive import (
    FDIRMode,
    FDIRExecutive,
)
from opengnc.fdir.sensor_fdir import (
    ChiSquareMonitor,
    GLRTMonitor,
    SensorHealth,
    SensorIsolationManager,
)


def test_chi_square_monitor_nominal():
    """Verify ChiSquareMonitor produces no alarms under nominal Gaussian residuals."""
    np.random.seed(42)
    dim = 3
    S = np.diag([0.01, 0.01, 0.01])
    monitor = ChiSquareMonitor(dim=dim, confidence=0.99, window_size=5, persistence_threshold=3)

    for _ in range(50):
        # Draw from N(0, S)
        r = np.random.multivariate_normal(np.zeros(dim), S)
        is_alarm, stat, thresh = monitor.update(r, S)
        assert not is_alarm
        assert thresh > 0.0


def test_chi_square_monitor_step_bias():
    """Verify ChiSquareMonitor flags persistent alarm upon large step bias injection."""
    dim = 3
    S = np.eye(3) * 0.01
    monitor = ChiSquareMonitor(dim=dim, confidence=0.99, window_size=5, persistence_threshold=3)

    # Inject 10-sigma bias
    large_bias = np.array([1.0, 0.0, 0.0])
    alarm_triggered = False

    for _ in range(10):
        is_alarm, stat, thresh = monitor.update(large_bias, S)
        if is_alarm:
            alarm_triggered = True
            break

    assert alarm_triggered


def test_glrt_monitor_slow_ramp():
    """Verify GLRT/CUSUM detects subtle slow drift ramp."""
    dim = 1
    glrt = GLRTMonitor(dim=dim, drift_allowance=0.1, alarm_threshold=3.0)

    is_fault = False
    for k in range(50):
        drift = 0.15 * k  # slow ramp
        is_fault, stats_val = glrt.update(np.array([drift]))
        if is_fault:
            break

    assert is_fault


def test_sensor_isolation_manager():
    """Verify SensorIsolationManager state transitions, covariance inflation, and consensus voting."""
    manager = SensorIsolationManager(
        ["ST1", "ST2", "GYRO1"], max_consecutive_faults=3, covariance_inflation_factor=10.0
    )

    # Initial state
    assert manager.sensors["ST1"] == SensorHealth.HEALTHY

    # Degraded transition
    manager.report_status("ST1", has_fault=True)
    assert manager.sensors["ST1"] == SensorHealth.DEGRADED

    # Isolated transition after 3 consecutive faults
    manager.report_status("ST1", has_fault=True)
    status = manager.report_status("ST1", has_fault=True)
    assert status == SensorHealth.ISOLATED

    # Covariance scaling check
    base_cov = np.eye(3) * 0.01
    healthy_cov = manager.get_effective_covariance("ST2", base_cov)
    assert np.allclose(healthy_cov, base_cov)

    isolated_cov = manager.get_effective_covariance("ST1", base_cov)
    assert np.all(np.diag(isolated_cov) >= 1e6)

    # Cross-consistency voting with 3 sensors where ST1 diverges
    measurements = {
        "S1": np.array([1.0, 0.0, 0.0]),
        "S2": np.array([0.9998, 0.01, 0.0]),
        "S3": np.array([0.7071, 0.7071, 0.0]),  # ~45 deg outlier
    }
    mgr = SensorIsolationManager(["S1", "S2", "S3"])
    outliers = mgr.cross_consistency_vote(measurements, angular_tol_deg=5.0)
    assert "S3" in outliers


def test_actuator_fault_detector():
    """Verify ActuatorFaultDetector flags speed overshoot and torque mismatch."""
    detector = ActuatorFaultDetector(
        max_wheel_speed=500.0, torque_tolerance=0.01, persistence_steps=2
    )

    # Nominal check
    status = detector.check_wheel("RW1", cmd_torque=0.05, actual_torque=0.05, current_speed=100.0)
    assert status == ActuatorHealth.HEALTHY

    # Torque mismatch
    detector.check_wheel("RW1", cmd_torque=0.05, actual_torque=0.01, current_speed=100.0)
    status = detector.check_wheel("RW1", cmd_torque=0.05, actual_torque=0.01, current_speed=100.0)
    assert status == ActuatorHealth.FAILED


def test_weighted_control_allocator_pyramid():
    """Verify 4-wheel pyramid control allocation and seamless 4-to-3 reallocation upon failure."""
    # Standard 4-wheel pyramid configuration at 30 deg inclination
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    B = np.array(
        [
            [c, 0.0, -c, 0.0],
            [0.0, c, 0.0, -c],
            [s, s, s, s],
        ]
    )

    allocator = WeightedControlAllocator(B, u_min=-0.1, u_max=0.1)
    tau_cmd = np.array([0.02, -0.01, 0.01])

    # 1. 4-Wheel Nominal Allocation
    u_4w = allocator.allocate(tau_cmd)
    assert len(u_4w) == 4
    assert np.allclose(B @ u_4w, tau_cmd, atol=1e-6)

    # 2. Wheel 0 Fails -> 3-Wheel Reallocation
    allocator.set_actuator_health(0, ActuatorHealth.FAILED)
    u_3w = allocator.allocate(tau_cmd)

    # Torque tracking must remain exact
    assert np.allclose(B @ u_3w, tau_cmd, atol=1e-6)
    # Failed wheel must receive zero torque command
    assert u_3w[0] == pytest.approx(0.0, abs=1e-8)


def test_underactuated_attitude_controller():
    """Verify 2-wheel underactuated controller stabilizes actuated axes with magnetic cross-damping."""
    controller = UnderactuatedAttitudeController(actuated_axes=(0, 1), kp=1.0, kd=2.0)

    q_err = np.array([0.05, -0.03, 0.02, 1.0])
    w_err = np.array([0.01, -0.01, 0.005])
    b_field = np.array([1e-5, 2e-5, 4e-5])  # Tesla

    tau_wheels, dipole = controller.compute_control(q_err, w_err, b_field)

    assert tau_wheels.shape == (3,)
    # Unactuated axis (index 2) must receive 0 wheel torque
    assert tau_wheels[2] == 0.0
    # Actuated axes receive restoring torque
    assert tau_wheels[0] < 0.0  # -kp*q[0] - kd*w[0]
    # Magnetic dipole produced for auxiliary damping
    assert np.linalg.norm(dipole) > 0.0


def test_fdir_executive_transitions():
    """Verify FDIRExecutive finite state machine transitions across nominal, degraded, underactuated, and safe modes."""
    exec_fsm = FDIRExecutive(initial_mode=FDIRMode.NOMINAL, max_rate_runaway=0.2)

    sensors = {"ST1": SensorHealth.HEALTHY, "GYRO1": SensorHealth.HEALTHY}
    actuators = {
        "RW1": ActuatorHealth.HEALTHY,
        "RW2": ActuatorHealth.HEALTHY,
        "RW3": ActuatorHealth.HEALTHY,
        "RW4": ActuatorHealth.HEALTHY,
    }

    # 1. Nominal check
    mode = exec_fsm.evaluate_state(0.0, sensors, actuators, pointing_error_deg=0.5, rate_norm=0.001)
    assert mode == FDIRMode.NOMINAL

    # 2. One wheel fails (3 remain) -> Degraded Pointing
    actuators["RW1"] = ActuatorHealth.FAILED
    mode = exec_fsm.evaluate_state(
        10.0, sensors, actuators, pointing_error_deg=0.5, rate_norm=0.001
    )
    assert mode == FDIRMode.DEGRADED_POINTING

    # 3. Second wheel fails (2 remain) -> Underactuated Control
    actuators["RW2"] = ActuatorHealth.FAILED
    mode = exec_fsm.evaluate_state(
        20.0, sensors, actuators, pointing_error_deg=0.5, rate_norm=0.001
    )
    assert mode == FDIRMode.UNDERACTUATED_CONTROL

    # 4. Angular rate runaway -> Safe Mode Sun Pointing
    mode = exec_fsm.evaluate_state(30.0, sensors, actuators, pointing_error_deg=5.0, rate_norm=0.3)
    assert mode == FDIRMode.SAFE_MODE_SUN_POINTING

    # Telemetry report
    telem = exec_fsm.get_telemetry()
    assert telem["current_mode"] == FDIRMode.SAFE_MODE_SUN_POINTING.value
    assert telem["transition_count"] >= 3
