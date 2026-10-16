import pytest
import numpy as np
from opengnc.propagators.kepler import KeplerPropagator
from opengnc.propagators.cowell import CowellPropagator
from opengnc.propagators.encke import EnckePropagator
from unittest.mock import patch


def test_kepler_circular_period():
    mu = 398600.4418e9
    r_mag = 7000e3
    v_circ = np.sqrt(mu / r_mag)

    r0 = np.array([r_mag, 0, 0])
    v0 = np.array([0, v_circ, 0])

    period = 2 * np.pi * np.sqrt(r_mag**3 / mu)

    propagator = KeplerPropagator(mu=mu)
    r_f, v_f = propagator.propagate(r0, v0, period)

    np.testing.assert_allclose(r_f, r0, atol=1e-5)
    np.testing.assert_allclose(v_f, v0, atol=1e-5)


def test_cowell_two_body_consistency():
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])

    dt = 100.0

    kepler_prop = KeplerPropagator(mu=mu)
    r_k, v_k = kepler_prop.propagate(r0, v0, dt)

    cowell_prop = CowellPropagator(mu=mu)
    r_c, v_c = cowell_prop.propagate(r0, v0, dt, dt_step=1.0)

    np.testing.assert_allclose(r_c, r_k, rtol=1e-5)
    np.testing.assert_allclose(v_c, v_k, rtol=1e-5)


def test_cowell_with_perturbation():
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])
    dt = 100.0

    def constant_thrust(t, r, v):
        return 0.01 * v / np.linalg.norm(v)

    cowell_prop = CowellPropagator(mu=mu)

    r_unp, _ = cowell_prop.propagate(r0, v0, dt, dt_step=1.0)

    r_pert, _ = cowell_prop.propagate(r0, v0, dt, perturbation_acc_fn=constant_thrust, dt_step=1.0)

    assert not np.allclose(r_unp, r_pert)


def test_cowell_short_step():
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])

    cowell_prop = CowellPropagator(mu=mu)
    r_f, v_f = cowell_prop.propagate(r0, v0, dt=5.0)  # dt < 10 triggers step_size = dt
    assert r_f is not None


def test_kepler_parabolic_and_hyperbolic():
    mu = 398600.4418e9
    r_mag = 7000e3

    prop = KeplerPropagator(mu=mu)

    v_par = np.sqrt(2 * mu / r_mag)
    r0 = np.array([r_mag, 0, 0])
    v0 = np.array([0, v_par, 0])
    r_f, v_f = prop.propagate(r0, v0, dt=100.0)
    assert r_f is not None

    v_hyp = 1.2 * v_par
    v0_hyp = np.array([0, v_hyp, 0])
    r_f, v_f = prop.propagate(r0, v0_hyp, dt=100.0)
    assert r_f is not None

    with patch("numpy.log", side_effect=Exception("Log Fail")):
        v0_hyp = np.array([0, v_hyp, 0])
        r_f, v_f = prop.propagate(r0, v0_hyp, dt=100.0)
        assert r_f is not None


def test_kepler_dt_wrap():
    mu = 398600.4418e9
    r_mag = 7000e3
    v_circ = np.sqrt(mu / r_mag)
    r0 = np.array([r_mag, 0, 0])
    v0 = np.array([0, v_circ, 0])

    prop = KeplerPropagator(mu=mu)
    period = 2 * np.pi * np.sqrt(r_mag**3 / mu)

    r_f, v_f = prop.propagate(r0, v0, dt=1.5 * period)
    assert r_f is not None


def test_encke_unperturbed():
    mu = 398600.4418e9  # m^3/s^2
    r_i = np.array([7000e3, 0.0, 0.0])
    v_i = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    dt = 1000.0  # 1000 seconds

    encke = EnckePropagator(mu=mu)
    kepler = KeplerPropagator(mu=mu)

    r_encke, v_encke = encke.propagate(r_i, v_i, dt)
    r_kep, v_kep = kepler.propagate(r_i, v_i, dt)

    assert np.allclose(r_encke, r_kep, rtol=1e-6)
    assert np.allclose(v_encke, v_kep, rtol=1e-6)


def test_encke_perturbed():
    mu = 398600.4418e9
    r_i = np.array([7000e3, 0.0, 0.0])
    v_i = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    dt = 100.0

    def constant_disturbance(t, r, v):
        return np.array([1e-6, 0.0, 0.0])  # 1 um/s^2

    encke = EnckePropagator(mu=mu, rect_tol=1e-12)  # Force rectification
    r_f, v_f = encke.propagate(r_i, v_i, dt, perturbation_acc_fn=constant_disturbance)

    assert len(r_f) == 3
    assert len(v_f) == 3
    kepler = KeplerPropagator(mu=mu)
    r_kep, _ = kepler.propagate(r_i, v_i, dt)

    assert not np.allclose(r_f, r_kep, rtol=0, atol=1e-6)


def test_encke_step_size_adjustments():
    mu = 398600.4418e9
    r_i = np.array([7000e3, 0.0, 0.0])
    v_i = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])

    encke = EnckePropagator(mu=mu)

    r_f, v_f = encke.propagate(r_i, v_i, dt=5.0, dt_step=10.0)
    assert r_f is not None

    r_f, v_f = encke.propagate(r_i, v_i, dt=15.0, dt_step=10.0)
    assert r_f is not None
