"""
Astrodynamics Benchmark Validation Suite.
Reference: David A. Vallado, Fundamentals of Astrodynamics and Applications (4th Edition).
NASA SP-8000 Series / Space Flight Mechanics Benchmark Vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengnc.guidance.maneuvers import bi_elliptic_transfer, hohmann_transfer
from opengnc.guidance.rendezvous import solve_lambert
from opengnc.utils.state_to_elements import eci2kepler, kepler2eci
from opengnc.utils.time_utils import calc_jd, jd_to_datetime


def test_vallado_example_2_5_state_to_elements_roundtrip():
    """
    Vallado (4th ed., Example 2-5, p. 117-118).
    Convert state vector to Keplerian elements and reconstruct Cartesian state.
    """
    # Canonical state vector (km -> m, km/s -> m/s)
    r_eci = np.array([6524.834, 6862.875, 6448.296]) * 1000.0
    v_eci = np.array([4.901327, 5.533756, -1.976341]) * 1000.0

    (
        a,
        ecc,
        incl,
        raan,
        argp,
        nu,
        m_anom,
        e_anom,
        p_val,
        arglat,
        truelon,
        lonper,
    ) = eci2kepler(r_eci, v_eci)

    # Vallado reference values (within numerical tolerances)
    assert a == pytest.approx(36127343.0, rel=1e-3)
    assert ecc == pytest.approx(0.832853, rel=1e-3)
    assert np.degrees(incl) == pytest.approx(87.87, abs=0.05)
    assert np.degrees(raan) == pytest.approx(227.90, abs=0.05)
    assert np.degrees(argp) == pytest.approx(53.38, abs=0.1)
    assert np.degrees(nu) == pytest.approx(92.34, abs=0.1)

    # Exact Cartesian reconstruction roundtrip test
    r_rec, v_rec = kepler2eci(a, ecc, incl, raan, argp, nu)
    assert np.allclose(r_rec, r_eci, atol=1e-2)
    assert np.allclose(v_rec, v_eci, atol=1e-4)


def test_vallado_example_3_5_julian_date():
    """
    Vallado (4th ed., Example 3-5, p. 183).
    Date: 2004-May-14 16:43:00 UTC.
    Expected JD: 2453140.196528.
    """
    jd_int, jd_frac = calc_jd(2004, 5, 14, 16, 43, 0.0)
    jd_total = jd_int + jd_frac
    assert jd_total == pytest.approx(2453140.196528, abs=1e-5)

    year, month, day, hour, minute, sec = jd_to_datetime(jd_int, jd_frac)
    assert (year, month, day, hour, minute) == (2004, 5, 14, 16, 43)
    assert sec == pytest.approx(0.0, abs=0.5)


def test_curtis_example_5_2_lamberts_problem():
    """
    Curtis (Orbital Mechanics for Engineering Students, Example 5-2, p. 257-260).
    Solving Lambert's problem for orbit targeting with time of flight 1 hour.
    """
    r1 = np.array([5000.0, 10000.0, 2100.0])  # km
    r2 = np.array([-14600.0, 2500.0, 7000.0])  # km
    tof = 3600.0  # s
    mu_earth = 398600.0  # km^3/s^2

    v1, v2 = solve_lambert(r1, r2, tof, mu=mu_earth, tm=1)

    # Curtis reference velocities (km/s)
    v1_ref = np.array([-5.9925, 1.9254, 3.2456])
    v2_ref = np.array([-3.3125, -4.1966, -0.3853])

    assert np.allclose(v1, v1_ref, atol=1e-3)
    assert np.allclose(v2, v2_ref, atol=1e-3)


def test_vallado_example_6_1_hohmann_transfer():
    """
    Vallado (4th ed., Example 6-1, p. 320).
    Hohmann transfer from LEO (r1 = 6578.137 km) to GEO (r2 = 42164.14 km).
    """
    r1 = 6578.137
    r2 = 42164.14
    mu = 398600.4418

    dv1, dv2, tof = hohmann_transfer(r1, r2, mu=mu)

    # Exact analytical reference values
    assert dv1 == pytest.approx(2.455, abs=0.01)
    assert dv2 == pytest.approx(1.478, abs=0.01)
    assert (dv1 + dv2) == pytest.approx(3.933, abs=0.01)
    assert tof == pytest.approx(18932.0, abs=10.0)  # ~ 5.26 hr


def test_vallado_example_6_2_bielliptic_transfer():
    """
    Vallado (4th ed., Example 6-2, p. 325).
    Bi-elliptic transfer with apoapsis rb = 105000 km.
    """
    r1 = 6578.137
    r2 = 42164.14
    rb = 105000.0
    mu = 398600.4418

    dv1, dv2, dv3, tof = bi_elliptic_transfer(r1, r2, rb, mu=mu)

    assert dv1 > 0.0
    assert dv2 > 0.0
    assert dv3 > 0.0
    assert tof > 18973.0  # Bi-elliptic TOF is larger than Hohmann
