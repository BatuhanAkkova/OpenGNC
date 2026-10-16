import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from opengnc.integrators import RK4, RK45, RK853, GaussJacksonIntegrator
from opengnc.integrators.ab_moulton import AdamsBashforthMoultonIntegrator
from opengnc.integrators.symplectic import SymplecticIntegrator
from opengnc.integrators.dop853_coeffs import A, C, B, E3, E5


@pytest.mark.parametrize("IntegratorClass", [RK4, RK45, RK853])
def test_linear_ode(IntegratorClass):
    if IntegratorClass == RK45:
        integrator = IntegratorClass(rtol=1e-8, atol=1e-10)
    elif IntegratorClass == RK853:
        integrator = IntegratorClass(rtol=1e-10, atol=1e-12)
    else:
        integrator = IntegratorClass()

    def f(t, y):
        return y

    t_span = (0, 1.0)
    y0 = [1.0]

    t_val, y_val = integrator.integrate(f, t_span, y0, dt=0.01)

    y_final = y_val[-1][0]
    y_expected = np.exp(1.0)

    if IntegratorClass == RK4:
        tol = 1e-5
    else:
        tol = 1e-6

    assert np.abs(y_final - y_expected) < tol


@pytest.mark.parametrize("IntegratorClass", [RK4, RK45, RK853])
def test_harmonic_oscillator(IntegratorClass):
    integrator = IntegratorClass()

    def f(t, x):
        return np.array([x[1], -x[0]])

    t_span = (0, 2 * np.pi)
    y0 = [0.0, 1.0]

    t_val, y_val = integrator.integrate(f, t_span, y0, dt=0.1)

    y_final = y_val[-1]
    y_expected = [0.0, 1.0]

    if IntegratorClass == RK4:
        tol = 1e-3
    else:
        tol = 1e-4

    np.testing.assert_allclose(y_final, y_expected, atol=tol)


def test_rk45_adaptive():
    integrator = RK45(rtol=1e-6, atol=1e-8)

    def f(t, y):
        return -y

    t_span = (0, 10.0)
    y0 = [1.0]

    t_val, _ = integrator.integrate(f, t_span, y0, dt=1.0)  # Start with large dt

    def f_steep(t, y):
        return 10 * y

    t_span = (0, 1.0)
    y0 = [1e-5]

    t_vals, _ = integrator.integrate(f_steep, t_span, y0, dt=0.1)
    dts = np.diff(t_vals)
    assert not np.allclose(dts, dts[0])


def test_rk853_precision():
    integrator = RK853(rtol=1e-12, atol=1e-12)

    def f(t, y):
        return y

    t_span = (0, 1.0)
    y0 = [1.0]

    _, y_val = integrator.integrate(f, t_span, y0, dt=0.1)

    y_final = y_val[-1][0]
    y_expected = np.exp(1.0)

    assert np.abs(y_final - y_expected) < 1e-11


def test_rk45_dt_none():
    integrator = RK45()

    def f(t, y):
        return y

    t_val, y_val = integrator.integrate(f, (0, 1.0), [1.0], dt=None)
    assert len(t_val) > 1


def test_rk853_dt_none():
    integrator = RK853()

    def f(t, y):
        return y

    t_val, y_val = integrator.integrate(f, (0, 1.0), [1.0], dt=None)
    assert len(t_val) > 1


def test_rk45_error_ratio_small():
    integrator = RK45(rtol=1e-3, atol=1e-3)

    def f(t, y):
        return np.zeros_like(y)

    y_next, t_next, dt_new = integrator.step(f, 0, np.array([1.0]), 0.1)
    assert dt_new == 0.1 * integrator.max_factor


def test_rk853_error_norm_zero():
    integrator = RK853()

    def f(t, y):
        return np.zeros_like(y)

    y_next, t_next, dt_new = integrator.step(f, 0, np.array([1.0]), 0.1)
    assert dt_new == 0.1 * integrator.max_factor


def test_rk45_step_too_small():
    integrator = RK45()

    def f_stiff(t, y):
        return 1e23 * y

    with pytest.raises(RuntimeError):
        # Call step directly to prevent infinite loop in integrate loop
        integrator.step(f_stiff, 0, np.array([1.0]), 0.1)


def test_rk853_step_too_small():
    integrator = RK853()

    def f_stiff(t, y):
        return 1e23 * y

    with pytest.raises(RuntimeError):
        integrator.step(f_stiff, 0, np.array([1.0]), 0.1)


def test_gauss_jackson_linear():
    gj_integrator = GaussJacksonIntegrator()

    def f_linear(t, y):
        return np.concatenate([y[3:], np.zeros(3)])

    r0 = np.array([100.0, 200.0, 300.0])
    v0 = np.array([1.0, 2.0, 3.0])
    y0 = np.concatenate([r0, v0])

    t_span = (0, 100.0)
    dt = 10.0

    t_values, y_values = gj_integrator.integrate(f_linear, t_span, y0, dt=dt)
    r_final = y_values[-1, :3]
    r_exact = r0 + 100.0 * v0
    error = np.linalg.norm(r_final - r_exact)
    assert error < 1e-10


def test_gauss_jackson_kepler():
    mu = 398600.4418e9  # m^3/s^2
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])

    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        return np.concatenate([v, -mu / (r_mag**3) * r])

    t_span = (0, 1000.0)
    dt = 10.0
    omega = np.sqrt(mu / 7000e3**3)
    gj_integrator = GaussJacksonIntegrator()
    t_values, y_values = gj_integrator.integrate(f, t_span, y0, dt=dt)

    t_end = t_span[1]
    r_exact = np.array([7000e3 * np.cos(omega * t_end), 7000e3 * np.sin(omega * t_end), 0.0])
    v_exact = np.array(
        [-7000e3 * omega * np.sin(omega * t_end), 7000e3 * omega * np.cos(omega * t_end), 0.0]
    )

    error_r = np.linalg.norm(y_values[-1, :3] - r_exact)
    error_v = np.linalg.norm(y_values[-1, 3:] - v_exact)
    assert error_r < 100000
    assert error_v < 200


def test_gauss_jackson_unsupported_step():
    gj_integrator = GaussJacksonIntegrator()
    with pytest.raises(NotImplementedError):
        gj_integrator.step(None, 0, None, 10.0)


def test_gauss_jackson_calc_differences_empty():
    gj_integrator = GaussJacksonIntegrator()
    diffs = gj_integrator._calc_differences([])
    assert np.allclose(diffs, np.zeros((1, 3)))


def test_gauss_jackson_boundary():
    def f(t, y):
        return np.array([0.0, 0, 0, 0, 0, 0])

    gj_integrator = GaussJacksonIntegrator()

    t_values, y_values = gj_integrator.integrate(f, (0, 5), np.zeros(6), dt=10)
    assert len(t_values) > 0

    t_values, y_values = gj_integrator.integrate(f, (0, 85), np.zeros(6), dt=10)
    assert np.isclose(t_values[-1], 85.0)


# --- Adams Bashforth Moulton Tests ---


def test_ab_moulton_kepler():
    """
    Test Adams-Bashforth-Moulton 8th order integrator on a Keplerian orbit.
    Compare with Analytical.
    """
    # Constants
    mu = 398600.4418e9  # m^3/s^2

    # Initial state (circular orbit at 7000 km)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])

    # Equations of motion
    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag**3) * r
        return np.concatenate([v, a])

    # Time span: ~1 orbit (approx 5800s)
    t_span = (0, 1000.0)
    dt = 10.0  # 10s step size
    omega = np.sqrt(mu / 7000e3**3)

    ab_integrator = AdamsBashforthMoultonIntegrator()
    t_values, y_values = ab_integrator.integrate(f, t_span, y0, dt=dt)

    # Analytical Solution at end time
    t_end = t_span[1]
    r_exact = np.array([7000e3 * np.cos(omega * t_end), 7000e3 * np.sin(omega * t_end), 0.0])
    v_exact = np.array(
        [-7000e3 * omega * np.sin(omega * t_end), 7000e3 * omega * np.cos(omega * t_end), 0.0]
    )

    # Compare final state with Exact
    r_ab_f = y_values[-1, :3]
    v_ab_f = y_values[-1, 3:]

    error_r = np.linalg.norm(r_ab_f - r_exact)
    error_v = np.linalg.norm(v_ab_f - v_exact)

    print(f"AB Position Error: {error_r} m")
    print(f"AB Velocity Error: {error_v} m/s")

    # Verify accurate to meters (expecting < 1e-3)
    assert error_r < 1e-3
    assert error_v < 1e-3


def test_ab_moulton_unsupported_step():
    ab_integrator = AdamsBashforthMoultonIntegrator()
    with pytest.raises(NotImplementedError):
        ab_integrator.step(None, 0, None, 10.0)


def test_ab_moulton_boundary():
    # Test h > (tf - t0)
    def f(t, y):
        return y

    t_values, y_values = AdamsBashforthMoultonIntegrator().integrate(
        f, (0, 5), np.array([1.0]), dt=10
    )
    assert len(t_values) > 0

    # Test curr_t + h > tf
    t_values, y_values = AdamsBashforthMoultonIntegrator().integrate(
        f, (0, 85), np.array([1.0]), dt=10
    )
    assert np.isclose(t_values[-1], 85.0)


# --- Dop853 Coeffs Tests ---


def test_dop853_array_sizes():
    assert C.shape == (16,)
    assert A.shape == (16, 16)
    assert B.shape == (12,)
    assert E3.shape == (13,)
    assert E5.shape == (13,)


def test_dop853_sum_rules():
    for i in range(12):
        row_sum = np.sum(A[i, :])
        assert np.isclose(row_sum, C[i], atol=1e-10)


# --- Symplectic Tests ---


def test_symplectic_kepler():
    mu = 398600.4418e9  # m^3/s^2
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])

    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag**3) * r
        return np.concatenate([v, a])

    t_span = (0, 1000.0)  # Time span: 1 orbit
    dt = 1.0  # Small step for accuracy

    sym_integrator = SymplecticIntegrator()
    t_values, y_values = sym_integrator.integrate(f, t_span, y0, dt=dt)

    assert len(t_values) > 1  # Check dimensions

    r_start = y_values[0, :3]
    v_start = y_values[0, 3:]
    E_start = np.sum(v_start**2) / 2 - mu / np.linalg.norm(r_start)

    r_end = y_values[-1, :3]
    v_end = y_values[-1, 3:]
    E_end = np.sum(v_end**2) / 2 - mu / np.linalg.norm(r_end)

    dE = np.abs(E_end - E_start) / np.abs(E_start)
    assert dE < 1e-6  # Highly conservative limit


def test_symplectic_unsupported_step():
    sym_integrator = SymplecticIntegrator()
    res_y, res_t, _ = sym_integrator.step(lambda t, y: y, 0, np.zeros(6), 1.0)
    assert len(res_y) == 6


def test_symplectic_boundary():
    def f(t, y):
        return np.zeros(6)

    sym_integrator = SymplecticIntegrator()
    t_values, y_values = sym_integrator.integrate(f, (0, 5), np.zeros(6), dt=10)
    assert len(t_values) > 0

    t_values, y_values = sym_integrator.integrate(f, (0, 15), np.zeros(6), dt=10)
    assert np.isclose(t_values[-1], 15.0)
