import numpy as np
import pytest
from opengnc.environment.multibody_dynamics import CR3BP


def test_cr3bp_jacobi_conservation():
    """Verify Jacobi constant is conserved in CR3BP."""
    # Earth-Moon system (mu ~ 0.01215)
    mu = 0.0121505856
    cr3bp = CR3BP(mu)

    # State: [x, y, z, vx, vy, vz]
    # L1 libration point approx for Earth-Moon
    L1_x = 0.836915  # Roughly
    state0 = np.array([L1_x, 0, 0, 0, 0.01, 0])

    c0 = cr3bp.calculate_jacobi_constant(state0)

    # Simple Euler integration to check derivative
    dt = 0.001
    derivs = cr3bp.get_dynamics(0, state0)
    state1 = state0 + derivs * dt

    c1 = cr3bp.calculate_jacobi_constant(state1)

    # Jacobi constant should be nearly identical for small dt
    assert np.isclose(c0, c1, atol=1e-6)


def test_cr3bp_l_points():
    """Check L4/L5 potential peaks."""
    mu = 0.01215
    cr3bp = CR3BP(mu)

    # L4 is at (0.5 - mu, sqrt(3)/2, 0)
    l4_x = 0.5 - mu
    l4_y = np.sqrt(3) / 2.0

    state_l4 = np.array([l4_x, l4_y, 0, 0, 0, 0])
    derivs = cr3bp.get_dynamics(0, state_l4)

    # Derivatives should be zero at equilibrium
    assert np.allclose(derivs[3:6], 0, atol=1e-3)


if __name__ == "__main__":
    test_cr3bp_jacobi_conservation()
    test_cr3bp_l_points()
