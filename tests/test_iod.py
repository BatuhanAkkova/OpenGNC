import numpy as np
import pytest
from opengnc.navigation.iod import (
    gibbs_iod,
    herrick_gibbs_iod,
    laplace_iod,
    laplace_iod_from_observations,
    gauss_iod,
    _stumpff,
    _kepler_U,
)
from opengnc.utils.state_to_elements import kepler2eci
from unittest.mock import patch


def solve_kepler(M, e):
    E = M
    for _ in range(10):
        f = E - e * np.sin(E) - M
        df = 1.0 - e * np.cos(E)
        E = E - f / df
        if abs(f) < 1e-12:
            break
    return E


def eccentric_to_true(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0))


def test_gauss_iod():
    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)

    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)

    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]

    rho_hats = []
    Rs = []

    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])

    expected_state_t2 = None

    for t in times:
        E = solve_kepler(n * t, ecc)
        nu = eccentric_to_true(E, ecc)
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)

        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)

        if abs(t - t2) < 1e-6:
            expected_state_t2 = np.concatenate([r, v])

    est_state = gauss_iod(
        rho_hats[0], rho_hats[1], rho_hats[2], times[0], times[1], times[2], Rs[0], Rs[1], Rs[2], mu
    )

    assert est_state[0:3] == pytest.approx(expected_state_t2[0:3], rel=3e-2)
    assert est_state[3:6] == pytest.approx(expected_state_t2[3:6], rel=5e-2)


def test_gibbs_iod():
    a = 7000e3
    ecc = 0.1
    incl = np.radians(28.5)
    raan = 0
    argp = 0

    taus = [0, 100, 200]
    positions = []
    actual_v2 = None

    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)

    for i, t in enumerate(taus):
        E = solve_kepler(n * t, ecc)
        nu = eccentric_to_true(E, ecc)
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        positions.append(r)
        if i == 1:
            actual_v2 = v

    v2_est = gibbs_iod(positions[0], positions[1], positions[2])

    assert v2_est == pytest.approx(actual_v2, rel=0.01)


def test_herrick_gibbs_iod():
    a = 7000e3
    ecc = 0.01
    incl = np.radians(28.5)
    raan = 0
    argp = 0

    dt = 10
    taus = [0, dt, 2 * dt]
    positions = []
    actual_v2 = None

    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)

    for i, t in enumerate(taus):
        E = solve_kepler(n * t, ecc)
        nu = eccentric_to_true(E, ecc)
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        positions.append(r)
        if i == 1:
            actual_v2 = v

    v2_est = herrick_gibbs_iod(positions[0], positions[1], positions[2], dt, dt)

    assert v2_est == pytest.approx(actual_v2, rel=1e-4)


def test_herrick_gibbs_iod_invalid_inputs():
    with pytest.raises(ValueError, match="Position vectors must be non-zero"):
        herrick_gibbs_iod(np.zeros(3), np.zeros(3), np.zeros(3), 10, 10)


def test_laplace_iod_from_observations():
    a = 7500e3
    ecc = 0.0
    incl = np.radians(45.0)
    raan = np.radians(30.0)
    argp = np.radians(60.0)

    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)

    t2 = 100.0
    dt = 10.0  # 10 seconds between observations
    times = [t2 - dt, t2, t2 + dt]

    rho_hats = []
    Rs = []

    R_fixed = np.array([6378e3, 1000e3, 500e3])

    expected_state_t2 = None

    for t in times:
        E = solve_kepler(n * t, ecc)
        nu = eccentric_to_true(E, ecc)
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)

        rho_vec = r - R_fixed
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_fixed)

        if t == t2:
            expected_state_t2 = np.concatenate([r, v])

    est_state = laplace_iod_from_observations(rho_hats, Rs, times, mu)

    assert est_state[0:3] == pytest.approx(expected_state_t2[0:3], rel=2e-5)
    assert est_state[3:6] == pytest.approx(expected_state_t2[3:6], rel=1e-4)


def test_iod_gibbs_singular():
    r = np.array([7000e3, 0, 0])
    v = gibbs_iod(r, r, r)
    assert np.all(v == 0)


def test_iod_gauss_singular():
    rho = np.array([1.0, 0, 0])
    R = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="LOS vectors are nearly coplanar"):
        gauss_iod(rho, rho, rho, 0, 10, 20, R, R, R)


def test_iod_laplace_singular_and_stumpff():
    rho = np.array([1.0, 0, 0])
    R = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="Determinant D is too small"):
        laplace_iod(rho, rho, rho, R, R, R)

    c2, c3 = _stumpff(-10)
    assert c2 is not None
    c2, c3 = _stumpff(0)
    assert c2 is not None


def test_gauss_iod_loop_solve_fail():
    rho_hat = np.array([1.0, 0, 0])
    R = np.array([0, 0, 0])
    sol_init = np.array([1.0, 1.0, 1.0])

    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    rho_hats = []
    Rs = []
    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])
    for t in times:
        f = t * n  # approx
        r, v = kepler2eci(a, ecc, incl, raan, argp, f)
        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)

    with patch("numpy.linalg.solve") as mock_solve:

        def solve_side_effect(mat, rhs):
            if solve_side_effect.count == 0:
                solve_side_effect.count += 1
                return np.array([1e6, 1e6, 1e6])
            else:
                raise np.linalg.LinAlgError("Singular")

        solve_side_effect.count = 0
        mock_solve.side_effect = solve_side_effect

        res = gauss_iod(
            rho_hats[0],
            rho_hats[1],
            rho_hats[2],
            times[0],
            times[1],
            times[2],
            Rs[0],
            Rs[1],
            Rs[2],
            mu,
        )
        assert res is not None


def test_gauss_iod_herrick_gibbs_fail():
    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    rho_hats = []
    Rs = []
    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])
    for t in times:
        f = t * n
        r, v = kepler2eci(a, ecc, incl, raan, argp, f)
        rho = r - R_obs
        rho_hats.append(rho / np.linalg.norm(rho))
        Rs.append(R_obs)

    with patch(
        "opengnc.navigation.iod._kepler_U", return_value=1.0
    ):  # mock loop execution or use real one
        with patch("opengnc.navigation.iod.herrick_gibbs_iod", side_effect=Exception("HG Fail")):
            res = gauss_iod(
                rho_hats[0],
                rho_hats[1],
                rho_hats[2],
                times[0],
                times[1],
                times[2],
                Rs[0],
                Rs[1],
                Rs[2],
                mu,
            )
            assert res is not None


def test_laplace_iod_no_roots():
    rho = np.array([1.0, 0, 0])
    dot = np.array([0, 1.0, 0])
    ddot = np.array([0, 0, 1.0])
    R = np.array([1000.0, 0, 0])
    R_dot = np.array([0, 0, 0])
    R_ddot = np.array([0, 0, 0])

    with patch("numpy.roots", return_value=np.array([-1.0, -2.0, -1.0j])):
        with pytest.raises(ValueError, match="No physical"):
            laplace_iod(rho, dot, ddot, R, R_dot, R_ddot)


def test_gauss_iod_divergence_break_low():
    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    rho_hats = []
    Rs = []
    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])
    for t in times:
        f = t * n
        r, v = kepler2eci(a, ecc, incl, raan, argp, f)
        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)

    with patch("numpy.linalg.norm", return_value=4000e3):
        res = gauss_iod(
            rho_hats[0],
            rho_hats[1],
            rho_hats[2],
            times[0],
            times[1],
            times[2],
            Rs[0],
            Rs[1],
            Rs[2],
            mu,
        )
        assert res is not None


def test_gauss_iod_divergence_break_high():
    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    rho_hats = []
    Rs = []
    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])
    for t in times:
        f = t * n
        r, v = kepler2eci(a, ecc, incl, raan, argp, f)
        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)

    with patch("numpy.linalg.norm", return_value=200000e3):
        res = gauss_iod(
            rho_hats[0],
            rho_hats[1],
            rho_hats[2],
            times[0],
            times[1],
            times[2],
            Rs[0],
            Rs[1],
            Rs[2],
            mu,
        )
        assert res is not None


def test_iod_stumpff_high_z():
    c2, c3 = _stumpff(1.0)
    assert c2 > 0 and c3 > 0
    assert np.isclose(c2, (1 - np.cos(1)) / 1.0)


def test_iod_kepler_U_no_converge():
    dt = 1e12
    r0 = 7000e3
    v0 = 7500.0
    mu = 3.986e14
    alpha = 2 / r0 - v0**2 / mu
    with patch("opengnc.navigation.iod._stumpff", return_value=(0.5, 0.16666)):
        chi = _kepler_U(dt, r0, v0, alpha, mu)
        assert chi is not None


def test_iod_kepler_U_converge():
    dt = 1000.0
    r0 = 7000e3
    v0 = 7500.0
    mu = 3.986e14
    alpha = 2 / r0 - v0**2 / mu
    chi = _kepler_U(dt, r0, v0, alpha, mu)
    assert chi > 0


def test_gauss_iod_no_physical_roots():
    with patch("numpy.roots", return_value=np.array([-1.0, -2.0])):
        with pytest.raises(ValueError, match="No physical radius solution found"):
            gauss_iod(
                np.eye(3)[0],
                np.eye(3)[1],
                np.eye(3)[2],
                0,
                1,
                2,
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            )
