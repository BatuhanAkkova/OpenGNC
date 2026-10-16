import pytest
import numpy as np
import os
from opengnc.disturbances.gravity import (
    TwoBodyGravity,
    J2Gravity,
    HarmonicsGravity,
    ThirdBodyGravity,
    RelativisticCorrection,
    OceanTidesGravity,
    GradientTorque,
)
from opengnc.disturbances.drag import LumpedDrag
from opengnc.disturbances.srp import Canonball
from opengnc.environment.density import Exponential
from opengnc.utils.time_utils import calc_jd
from unittest.mock import MagicMock

MU = 398600.4418e9  # m^3/s^2


@pytest.fixture
def ephemeris():
    r_eci = np.array([7000e3, 0.0, 0.0])  # 7000 km altitude on X-axis
    v_eci = np.array([0.0, 7.5e3, 0.0])  # Circular velocity
    jd = 2460337.5  # Arbitrary date
    return r_eci, v_eci, jd


@pytest.fixture
def state():
    r_eci = np.array([7000e3, 0.0, 0.0])  # meters
    jd = 2460337.5  # Arbitrary date
    return r_eci, jd


def test_two_body_gravity(ephemeris):
    r_eci, _, _ = ephemeris
    model = TwoBodyGravity(mu=MU)
    acc = model.get_acceleration(r_eci)

    r_norm = np.linalg.norm(r_eci)
    expected_acc = -MU / r_norm**3 * r_eci

    np.testing.assert_allclose(acc, expected_acc, rtol=1e-5)


def test_j2_gravity_equator(ephemeris):
    r_eci, _, _ = ephemeris
    model = J2Gravity(mu=MU, j2=0.001082635855)
    acc = model.get_acceleration(r_eci)
    assert abs(acc[2]) < 1e-9


def test_j2_gravity_polar():
    r_eci = np.array([0.0, 0.0, 7000e3])
    model = J2Gravity(mu=MU, j2=0.001082635855)
    acc = model.get_acceleration(r_eci)
    two_body = TwoBodyGravity(mu=MU).get_acceleration(r_eci)
    assert not np.allclose(acc, two_body)


def test_harmonics_gravity_loading():
    model = HarmonicsGravity(mu=MU, n_max=2, m_max=2)
    assert np.any(model.C != 0) or np.any(model.S != 0)


def test_drag_opposes_velocity(ephemeris):
    r_eci, v_eci, jd = ephemeris
    density_model = Exponential(rho0=1e-12, H=1000.0)
    model = LumpedDrag(density_model)

    mass = 100.0
    area = 1.0
    cd = 2.2

    acc = model.get_acceleration(r_eci, v_eci, jd, mass, area, cd)
    assert np.dot(acc, v_eci) < 0


def test_srp_with_mocked_sun(mocker, ephemeris):
    r_eci, _, jd = ephemeris
    mock_sun = mocker.patch("opengnc.environment.solar.Sun.calculate_sun_eci")
    mock_sun.return_value = np.array([1.496e11, 0, 0])

    model = Canonball()
    mass = 100.0
    area = 1.0
    cr = 1.0

    acc = model.get_acceleration(r_eci, jd, mass, area, cr)
    assert acc.shape == (3,)
    assert acc[0] < 0


def test_drag_with_co_rotation(ephemeris):
    r_eci, v_eci, jd = ephemeris
    density_model = Exponential(rho0=1e-12, H=1000.0)
    model = LumpedDrag(density_model, co_rotate=True)

    mass = 100.0
    area = 1.0
    cd = 2.2

    acc = model.get_acceleration(r_eci, v_eci, jd, mass, area, cd)
    assert np.dot(acc, v_eci) < 0

    model_no_rot = LumpedDrag(density_model, co_rotate=False)
    acc_no_rot = model_no_rot.get_acceleration(r_eci, v_eci, jd, mass, area, cd)

    assert not np.allclose(acc, acc_no_rot)


def test_drag_low_velocity():
    mock_density = MagicMock()
    mock_density.get_density.return_value = 1.0
    drag = LumpedDrag(mock_density)
    r = np.array([7000e3, 0, 0])
    w_earth = np.array([0, 0, 7.2921159e-5])
    v = np.cross(w_earth, r)
    acc = drag.get_acceleration(r, v, 2460000.5, 1000, 1, 2.2)
    assert np.allclose(acc, 0.0)


def test_srp_eclipse(ephemeris):
    r_eci, _, jd = ephemeris
    model = Canonball()

    r_sun = model.sun_model.calculate_sun_eci(jd)
    u_sun = r_sun / np.linalg.norm(r_sun)

    r_sat_shadow = -u_sun * 7000e3

    acc = model.get_acceleration(r_sat_shadow, jd, 100.0, 1.0, 1.0)
    assert np.allclose(acc, 0.0)

    r_perp = np.array([-u_sun[1], u_sun[0], 0])
    if np.linalg.norm(r_perp) < 1e-9:
        r_perp = np.cross(u_sun, np.array([1.0, 0.0, 0.0]))
    r_perp = r_perp / np.linalg.norm(r_perp)

    r_sat_no_shadow = -u_sun * 7000e3 + r_perp * 7000e3

    acc_no_shadow = model.get_acceleration(r_sat_no_shadow, jd, 100.0, 1.0, 1.0)
    assert np.linalg.norm(acc_no_shadow) > 0


def test_harmonics_gravity_file_not_found(capsys):
    model = HarmonicsGravity(file_path="non_existent_file.csv")
    captured = capsys.readouterr()
    assert "Warning: Gravity coefficient file not found" in captured.out
    assert np.all(model.C == 0)


def test_third_body_gravity():
    model = ThirdBodyGravity()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    jd = 2451545.0 + 0.5

    acc = model.get_acceleration(r_eci, jd)
    assert np.linalg.norm(acc) > 0
    assert np.linalg.norm(acc) < 1e-5


def test_relativistic_correction():
    model = RelativisticCorrection()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    v_eci = np.array([0.0, 7.5, 0.0]) * 1000.0

    acc = model.get_acceleration(r_eci, v_eci)
    assert np.linalg.norm(acc) > 0
    assert np.linalg.norm(acc) < 1e-6


def test_harmonics_gravity_acceleration():
    model = HarmonicsGravity(mu=398600.4418e9, n_max=3, m_max=3)

    model.C[2, 0] = -1.082e-3
    model.C[2, 2] = 1.57e-6
    model.S[2, 2] = -0.9e-6

    r_eci = np.array([7000e3, 0.0, 0.0])
    jd = 2451545.0

    acc = model.get_acceleration(r_eci, jd)
    assert acc.shape == (3,)
    assert np.linalg.norm(acc) > 0


def test_relativistic_correction_with_j_earth():
    J_earth = np.array([0, 0, 1e12])
    model = RelativisticCorrection(J_earth=J_earth)
    r_eci = np.array([7000e3, 0, 0])
    v_eci = np.array([0, 7500, 0])

    acc = model.get_acceleration(r_eci, v_eci)
    assert np.linalg.norm(acc) > 0


def test_ocean_tides_gravity_pole():
    model = OceanTidesGravity()
    r_eci = np.array([0.0, 0.0, 7000e3])
    jd = 2451545.0

    acc = model.get_acceleration(r_eci, jd)
    assert acc.shape == (3,)
    assert np.all(np.isfinite(acc))


def test_harmonics_gravity_acceleration_pole():
    model = HarmonicsGravity(mu=398600.4418e9, n_max=3, m_max=3)
    r_eci = np.array([0.0, 0.0, 7000e3])
    jd = 2451545.0

    acc = model.get_acceleration(r_eci, jd)
    assert acc.shape == (3,)
    assert np.all(np.isfinite(acc))


def test_ocean_tides_acceleration_non_zero(state):
    r_eci, jd = state
    model = OceanTidesGravity()
    acc = model.get_acceleration(r_eci, jd)

    assert np.any(acc != 0.0)

    mag = np.linalg.norm(acc)
    assert mag < 1e-6
    assert mag > 1e-12  # Should be at least something


def test_ocean_tides_time_varying(state):
    r_eci, jd = state
    model = OceanTidesGravity()
    acc1 = model.get_acceleration(r_eci, jd)
    acc2 = model.get_acceleration(r_eci, jd + 0.5)  # 12 hours later

    assert not np.allclose(acc1, acc2, atol=1e-15, rtol=1e-12)


def test_ocean_tides_altitude_dependence():
    model = OceanTidesGravity()
    acc_close = model.get_acceleration(np.array([7000e3, 0, 0]), 2460337.5)
    acc_far = model.get_acceleration(np.array([42000e3, 0, 0]), 2460337.5)

    mag_close = np.linalg.norm(acc_close)
    mag_far = np.linalg.norm(acc_far)

    assert mag_far < mag_close


def test_gravity_gradient_torque():
    model = GradientTorque()
    J = np.diag([100.0, 200.0, 300.0])
    r_eci = np.array([7000e3, 0.0, 0.0])
    theta = np.pi / 4
    q_body2eci = np.array([0.0, np.sin(theta / 2), 0.0, np.cos(theta / 2)])

    t_gg = model.gravity_gradient_torque(J, r_eci, q_body2eci)
    assert t_gg.shape == (3,)
    assert np.any(t_gg != 0.0)

    t_gg_zero = model.gravity_gradient_torque(J, np.zeros(3), q_body2eci)
    assert np.allclose(t_gg_zero, 0.0)
