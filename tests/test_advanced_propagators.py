import numpy as np
import pytest
from opengnc.propagators.gve import GVEPropagator
from opengnc.integrators.taylor import TaylorIntegrator


def test_gve_propagation():
    """Verify GVE basic orbital element changes."""
    mu = 3.986e14
    gve = GVEPropagator(mu=mu)

    # $[a, e, i, \Omega, \omega, \theta]$
    state0 = np.array([7000e3, 0.001, np.radians(28.5), 0, 0, 0])

    # 1 m/s^2 radial acceleration
    def radial_acc(t, s):
        return np.array([1.0, 0.0, 0.0])

    dt = 10.0
    state1 = gve.propagate(0, state0, dt, radial_acc)

    # Semi-major axis should change
    assert not np.allclose(state0, state1)
    assert not np.isnan(state1).any()


def test_taylor_integrator():
    """Verify Taylor series integrator on simple harmonic oscillator."""
    ti = TaylorIntegrator(order=2)

    # y'' = -y -> y' = v, v' = -y
    def f(t, state):
        y, v = state
        return np.array([v, -y])

    y0 = np.array([1.0, 0.0])
    dt = 0.1
    y1, t1, dt_s = ti.step(f, 0, y0, dt)

    # Exact solution: cos(t)
    assert np.isclose(y1[0], np.cos(0.1), atol=1e-3)


if __name__ == "__main__":
    test_gve_propagation()
    test_taylor_integrator()
