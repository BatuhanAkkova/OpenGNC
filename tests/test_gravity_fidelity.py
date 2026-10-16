import time
import numpy as np
import pytest
from opengnc.disturbances.gravity import HarmonicsGravity


def test_harmonics_gravity_fidelity():
    """Verify optimized harmonics gravity runs and returns valid acceleration."""
    # Earth parameters
    mu = 398600.4418e9
    re = 6378137.0

    # Test high degree/order
    n_max, m_max = 50, 50
    gravity = HarmonicsGravity(mu=mu, re=re, n_max=n_max, m_max=m_max)

    r_eci = np.array([7000e3, 0, 0])  # 7000 km altitude (approx)
    jd = 2451545.0  # J2000

    # Warm up numba
    acc1 = gravity.get_acceleration(r_eci, jd)

    # Measure time
    start = time.time()
    for _ in range(10):
        acc = gravity.get_acceleration(r_eci, jd)
    end = time.time()

    avg_time = (end - start) / 10.0
    print(f"Average time for degree {n_max}: {avg_time:.6f} s")
    print(f"Acceleration vector: {acc}")
    print(f"Acceleration norm: {np.linalg.norm(acc)}")

    assert len(acc) == 3
    assert np.all(np.isfinite(acc))
    # Two-body approx: mu / r^2 = 3.98e14 / (7e6)^2 = 8.12
    assert np.linalg.norm(acc) > 8.0 and np.linalg.norm(acc) < 8.5


if __name__ == "__main__":
    test_harmonics_gravity_fidelity()
