import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from opengnc.navigation import (
    OrbitDeterminationEKF,
    AngleOnlyNavigation,
    GPSNavigation,
    RelativeNavigationEKF,
    SurfaceNavigationEKF,
)
from opengnc.navigation.terrain_nav import FeatureMatchingTRN, map_relative_localization_update
from opengnc.navigation.batch_ls import BatchLeastSquaresOD
from opengnc.navigation.iod import gauss_iod
from opengnc.utils.state_to_elements import kepler2eci


def test_orbit_determination_ekf():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])

    P0 = np.eye(6) * 100.0
    Q = np.eye(6) * 1.0  # Process noise
    R = np.eye(3) * 10.0  # Pos measurement noise

    od = OrbitDeterminationEKF(x0, P0, Q, R, use_j2=False)

    dt = 1.0
    od.predict(dt)

    assert od.state.shape == (6,)
    assert od.state[1] > 0.0

    z_meas = r0 + np.array([5.0, -2.0, 1.0])
    od.update(z_meas)

    assert np.all(np.diag(od.covariance) >= 0)


def test_angle_only_navigation():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])

    P0 = np.eye(6) * 1000.0
    Q = np.eye(6) * 1.0
    R = np.eye(3) * 0.001  # Angle noise (in unit vector space)

    nav = AngleOnlyNavigation(x0, P0, Q, R)

    target_pos = np.array([0, 0, 0])
    u_meas = -r0 / np.linalg.norm(r0)  # True LOS

    nav.update_unit_vector(u_meas, target_pos)
    assert nav.state.shape == (6,)


def test_gps_navigation():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])
    P0 = np.eye(6) * 100.0
    Q = np.eye(6) * 1.0
    R = np.eye(3) * 5.0

    nav = GPSNavigation(x0, P0, Q, R)

    r_meas = r0 + np.random.normal(0, 5, 3)
    v_meas = v0 + np.random.normal(0, 0.1, 3)

    nav.update_gps(r_meas, v_meas)
    assert nav.state.shape == (6,)


def test_relative_navigation():
    n = 0.0011  # LEO mean motion
    x0 = np.array([100.0, 0, 0, 0, -0.1, 0])  # 100m radial separation
    P0 = np.eye(6) * 10.0
    Q = np.eye(6) * 0.1
    R = np.eye(3) * 1.0

    nav = RelativeNavigationEKF(x0, P0, Q, R, n)

    nav.predict(dt=10.0)
    assert nav.state.shape == (6,)

    nav.update(np.array([101.0, 0.5, 0.1]))
    assert nav.state.shape == (6,)


def test_surface_navigation():
    x0 = np.zeros(6)
    P0 = np.eye(6) * 10.0
    Q = np.eye(6) * 0.1
    R = np.eye(3) * 0.1

    nav = SurfaceNavigationEKF(x0, P0, Q, R)

    landmark = np.array([10, 0, 0])
    z_obs = landmark - np.array([0.1, 0, 0])  # Measured from slighly offset pos

    nav.predict(dt=1.0, accel=np.array([1.0, 0, 0]))
    nav.update_landmark(z_obs, landmark)

    assert nav.state.shape == (6,)


def test_terrain_relative_navigation():
    map_db = [np.array([10.0, 0.0, 0.0]), np.array([20.0, 5.0, 0.0])]
    trn = FeatureMatchingTRN(map_db)

    obs = [np.array([10.1, 0.2, 0.0])]
    matches = trn.match_features(obs)
    assert len(matches) == 1
    assert np.array_equal(matches[0][0], map_db[0])

    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    P = np.eye(6) * 10.0
    R = np.eye(3) * 0.1

    x_new, P_new = map_relative_localization_update(x, P, matches, R)
    assert x_new.shape == (6,)
    assert P_new.shape == (6, 6)
    assert not np.array_equal(x, x_new)  # Should update state
    assert np.all(np.diag(P_new)[:3] < np.diag(P)[:3])  # Should decrease position uncertainty


def test_batch_ls_singular():
    obs = [np.array([100, 0, 0]), np.array([100, 0, 0])]
    times = [0, 0]
    sol = BatchLeastSquaresOD([7000e3, 0, 0, 0, 7.5e3, 0])
    sol.solve(obs, times, max_iter=1)


def test_gps_custom_r_and_pos_only():
    gps = GPSNavigation(np.zeros(6), np.eye(6), np.eye(6), np.eye(3))
    gps.ekf = MagicMock()
    gps.update_gps(np.zeros(3), np.zeros(3), R_gps=np.eye(6))
    assert gps.ekf.update.called
    gps.ekf.reset_mock()
    gps.update_gps(np.zeros(3))


def test_terrain_nav_no_matches():
    x = np.zeros(6)
    P = np.eye(6)
    updated_x, updated_P = map_relative_localization_update(x, P, [], np.eye(3))
    assert np.all(updated_x == x)


def test_angle_only_singular():
    nav = AngleOnlyNavigation(np.zeros(6), np.eye(6), np.eye(6), np.eye(3))
    nav.ekf = MagicMock()
    nav.update_unit_vector(np.array([1, 0, 0]), np.zeros(3))
    assert True


def test_angle_only_navigation_near_zero_range():
    nav = AngleOnlyNavigation(np.zeros(6), np.eye(6), np.eye(6), np.eye(3))
    target = np.array([0.0, 0.0, 0.0])  # same as state[:3]
    u_meas = np.array([1.0, 0.0, 0.0])
    nav.update_unit_vector(u_meas, target)
    assert True


def test_gauss_iod_no_positive_roots():
    with patch("numpy.roots", return_value=np.array([-1.0, -2.0])):
        rho1 = np.array([1.0, 0, 0])
        rho2 = np.array([0, 1.0, 0])
        rho3 = np.array([0, 0, 1.0])
        R = np.array([7000e3, 0, 0])
        with pytest.raises(ValueError, match="No physical"):
            gauss_iod(rho1, rho2, rho3, 0, 10, 20, R, R, R)


def test_gauss_iod_radius_out_of_bounds():
    rho = np.array([1, 0, 0])
    R1 = np.array([1000e3, 0, 0])
    R2 = np.array([0, 1000e3, 0])
    R3 = np.array([0, 0, 1000e3])
    rho1 = np.array([1.0, 0, 0])
    rho2 = np.array([0, 1.0, 0])
    rho3 = np.array([0, 0, 1.0])

    with patch("numpy.roots", return_value=np.array([200000000e3])):
        with pytest.raises(ValueError, match="Radius out of bounds"):
            gauss_iod(rho1, rho2, rho3, 0, 10, 20, R1, R2, R3)


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


def test_batch_ls_od():
    a = 7000e3
    ecc = 0.001
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(10)
    nu = 0

    x_true, _ = kepler2eci(a, ecc, incl, raan, argp, nu)
    v_true = np.array([0, 7500, 0])  # Simple circular-ish velocity
    _, v_true = kepler2eci(a, ecc, incl, raan, argp, nu)
    x0_true = np.concatenate([x_true, v_true])

    times = [0, 10, 20, 30, 40, 50]
    observations = []
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)

    for t in times:
        r_t, _ = kepler2eci(
            a, ecc, incl, raan, argp, eccentric_to_true(solve_kepler(nu + n * t, ecc), ecc)
        )
        observations.append(r_t + np.random.normal(0, 1.0, 3))

    x_guess = x0_true + np.array([1000, -1000, 500, 1, -1, 0.5])

    bls = BatchLeastSquaresOD(x_guess)
    x_est = bls.solve(observations, times)

    assert x_est[:3] == pytest.approx(x0_true[:3], rel=1e-3)
    assert x_est[3:] == pytest.approx(x0_true[3:], rel=1e-3)
