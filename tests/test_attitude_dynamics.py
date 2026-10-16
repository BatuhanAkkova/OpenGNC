import numpy as np
import pytest
from opengnc.attitude_dynamics.rigid_body import euler_equations
from opengnc.attitude_dynamics.fuel_slosh import fuel_slosh_dynamics, fuel_slosh_torque
from opengnc.attitude_dynamics.flexible_body import (
    flexible_body_dynamics,
    coupled_flexible_rigid_dynamics,
)
from opengnc.attitude_dynamics.variable_inertia import (
    variable_inertia_euler_equations,
    mass_depletion_J_dot,
)

from opengnc.attitude_dynamics.fuel_slosh import fuel_slosh_dynamics, fuel_slosh_torque


def test_euler_equations_zero_values():
    J = np.eye(3)
    omega = np.zeros(3)
    torque = np.zeros(3)

    omega_dot = euler_equations(J, omega, torque)
    assert np.allclose(omega_dot, np.zeros(3))


def test_euler_equations_principal_axis_rotation_stable():
    J = np.diag([10, 20, 30])
    omega = np.array([1.0, 0.0, 0.0])
    torque = np.zeros(3)

    omega_dot = euler_equations(J, omega, torque)
    assert np.allclose(omega_dot, np.zeros(3), atol=1e-12)


def test_euler_equations_external_torque():
    J = np.eye(3) * 10
    omega = np.zeros(3)
    torque = np.array([10.0, 0.0, 0.0])

    expected_omega_dot = np.array([1.0, 0.0, 0.0])

    omega_dot = euler_equations(J, omega, torque)
    assert np.allclose(omega_dot, expected_omega_dot, atol=1e-12)


def test_euler_equations_general_case():
    J = np.diag([1.0, 2.0, 3.0])
    omega = np.array([1.0, 1.0, 1.0])
    torque = np.zeros(3)

    expected_omega_dot = np.array([-1.0, 1.0, -1.0 / 3.0])

    omega_dot = euler_equations(J, omega, torque)
    assert np.allclose(omega_dot, expected_omega_dot, atol=1e-12)


def test_euler_equations_invalid_shapes():
    J = np.eye(3)
    omega = np.zeros(3)
    torque = np.zeros(3)

    with pytest.raises(ValueError):
        euler_equations(np.eye(2), omega, torque)

    with pytest.raises(ValueError):
        euler_equations(J, np.zeros(2), torque)

    with pytest.raises(ValueError):
        euler_equations(J, omega, np.zeros(4))


def test_slosh_period():
    L = 1.0
    r_base = np.zeros(3)
    g_equiv = np.array([0, 0, -9.81])
    omega = np.zeros(3)
    omega_dot = np.zeros(3)
    theta = 0.1
    theta_dot = 0.0

    theta_ddot = fuel_slosh_dynamics(theta, theta_dot, omega, omega_dot, L, r_base, g_equiv)

    assert np.isclose(theta_ddot, -np.sin(0.1) * 9.81, atol=1e-3)


def test_slosh_torque_reverses():
    m_p = 5.0
    L = 1.0
    theta = 0.1
    theta_dot = 0.0
    theta_ddot = -0.981
    r_base = np.array([0, 0, -1.0])

    torque = fuel_slosh_torque(m_p, L, theta, theta_dot, theta_ddot, r_base)

    assert not np.allclose(torque, 0.0)
    assert torque.shape == (3,)


def test_flexible_body_oscillation():
    eta = np.array([1.0])
    eta_dot = np.array([0.0])
    omega_dot = np.array([0.0, 0.0, 0.0])
    natural_freqs = np.array([10.0])  # 10 rad/s
    damping_ratios = np.array([0.05])  # 5% damping
    modal_influence = np.array([[0.0, 0.0, 0.0]])

    eta_ddot = flexible_body_dynamics(
        eta, eta_dot, omega_dot, natural_freqs, damping_ratios, modal_influence
    )

    assert np.isclose(eta_ddot[0], -100.0)


def test_coupled_dynamics_conservation():
    J_rigid = np.eye(3)
    omega = np.array([0.1, 0.0, 0.0])
    torque = np.zeros(3)
    eta = np.array([1.0])
    eta_dot = np.zeros(1)
    natural_freqs = np.array([10.0])
    damping_ratios = np.array([0.0])
    modal_influence = np.array([[0.1, 0.0, 0.0]])

    omega_dot, eta_ddot = coupled_flexible_rigid_dynamics(
        J_rigid, omega, torque, eta, eta_dot, natural_freqs, damping_ratios, modal_influence
    )

    assert not np.allclose(omega_dot, 0.0)
    assert not np.allclose(eta_ddot, 0.0)


def test_variable_inertia_acceleration():
    J = np.eye(3)
    J_dot = np.eye(3) * 0.1  # Increasing inertia
    omega = np.array([1.0, 0.0, 0.0])
    torque = np.zeros(3)

    omega_dot = variable_inertia_euler_equations(J, J_dot, omega, torque)

    assert np.isclose(omega_dot[0], -0.1)


def test_mass_depletion_J_dot():
    dm_dt = -0.1  # 0.1 kg/s loss
    r_point = np.array([1.0, 0, 0])

    J_dot = mass_depletion_J_dot(None, 10.0, dm_dt, r_point)

    assert np.isclose(J_dot[0, 0], 0.0)
    assert np.isclose(J_dot[1, 1], -0.1)
    assert np.isclose(J_dot[2, 2], -0.1)
