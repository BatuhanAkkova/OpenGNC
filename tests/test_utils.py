import pytest
import numpy as np
from opengnc.utils.quat_utils import (
    quat_normalize,
    quat_conj,
    quat_norm,
    quat_inv,
    quat_mult,
    quat_rot,
    quat_to_rmat,
    axis_angle_to_quat,
)
from opengnc.utils.time_utils import (
    calc_jd,
    jd_to_datetime,
    day_to_mdtime,
    calc_gmst,
    calc_last,
    calc_lst,
    calc_doy,
    is_leap_year,
    convert_time,
)
from opengnc.utils.state_to_elements import eci2kepler, kepler2eci, anomalies
from opengnc.utils.frame_conversion import (
    eci2ecef,
    ecef2eci,
    eci2lvlh_dcm,
    eci2llh,
    elements2perifocal_dcm,
)
from opengnc.utils.state_conversion import (
    quat_to_dcm,
    quat_to_euler,
    dcm_to_quat,
    dcm_to_euler,
    euler_to_quat,
    euler_to_dcm,
    rot_x,
    rot_y,
    rot_z,
)
from opengnc.utils.mean_elements import osculating2mean, get_j2_secular_rates
from opengnc.utils.mrp_utils import quat_to_mrp


# MEAN ELEMENTS
def test_osculating2mean_identity():
    elements = (7000e3, 0.01, 1.0, 0.0, 0.0, 0.0)
    out = osculating2mean(*elements)
    assert out == elements


def test_j2_secular_rates():
    a = 7000e3
    ecc = 0.001
    incl = np.radians(45.0)

    raan_dot, argp_dot, M_dot = get_j2_secular_rates(a, ecc, incl)
    assert raan_dot < 0
    assert argp_dot > 0
    n = np.sqrt(398600.4415e9 / a**3)
    assert M_dot > n


# QUAT UTILS


def test_quat_norm_normalize():
    q = np.array([3, 0, 4, 0])  # Norm is 5
    assert np.isclose(quat_norm(q), 5.0)

    q_norm = quat_normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)
    assert np.allclose(q_norm, np.array([0.6, 0, 0.8, 0]))

    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))


def test_quat_ops_basic():
    q = np.array([0, 0, 0, 1])  # Identity

    q_conj = quat_conj(q)  # Conjugate
    np.testing.assert_array_equal(q_conj, np.array([0, 0, 0, 1]))

    q_inv = quat_inv(q)  # Inverse
    np.testing.assert_array_equal(q_inv, np.array([0, 0, 0, 1]))

    q2 = np.array([1, 0, 0, 0])
    q_mult = quat_mult(q2, q2)
    np.testing.assert_array_equal(q_mult, np.array([0, 0, 0, -1]))


def test_quat_rot():
    angle = np.pi / 2
    q = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])

    v = np.array([1, 0, 0])
    v_rot = quat_rot(q, v)

    np.testing.assert_allclose(v_rot, np.array([0, 1, 0]), atol=1e-10)


def test_axis_angle_to_quat():
    axis = np.array([0, 0, 1], dtype=float)
    angle = np.pi / 2
    axis_angle = axis * angle

    q = axis_angle_to_quat(axis_angle)
    expected = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    np.testing.assert_allclose(q, expected, atol=1e-10)

    q_zero = axis_angle_to_quat(np.zeros(3))  # Zero rotation
    np.testing.assert_array_equal(q_zero, np.array([0, 0, 0, 1]))


def test_quat_to_rmat():
    q = np.array([0, 0, 0, 1])  # Identity
    rmat = quat_to_rmat(q)
    np.testing.assert_array_equal(rmat, np.eye(3))

    val = 1.0 / np.sqrt(2)
    q_z = np.array([0, 0, val, val])  # 90 deg Z rotation
    rmat_z = quat_to_rmat(q_z)

    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rmat_z, expected, atol=1e-10)


# TIME UTILS


def test_calc_jd():
    # J2000 Epoch: 2000 Jan 1 12:00:00 TT -> JD 2451545.0
    jd, jdfrac = calc_jd(2000, 1, 1, 12, 0, 0)
    assert jd + jdfrac == 2451545.0


def test_jd_datetime_roundtrip():
    y, m, d, h, mn, s = 2024, 1, 18, 10, 30, 45.0
    jd, jdfrac = calc_jd(y, m, d, h, mn, s)
    y2, m2, d2, h2, mn2, s2 = jd_to_datetime(jd, jdfrac)

    assert y == y2
    assert m == m2
    assert d == d2
    assert h == h2
    assert mn == mn2
    assert np.isclose(s, s2, atol=1e-6)


def test_calc_gmst():
    jd = 2451545.0
    gmst_rad = calc_gmst(jd)
    gmst_deg = np.degrees(gmst_rad)

    assert np.isclose(gmst_deg, 280.4606, atol=1e-2)  # Expected approx 280.4606 deg

    assert 0 <= gmst_rad < 2 * np.pi  # Range check


def test_is_leap_year():
    assert is_leap_year(2000)
    assert is_leap_year(2024)
    assert not is_leap_year(2100)
    assert not is_leap_year(2023)


def test_calc_doy():
    assert calc_doy(2024, 1, 1) == 1
    assert calc_doy(2024, 2, 29) == 60
    assert calc_doy(2023, 2, 28) == 59
    assert calc_doy(2023, 3, 1) == 60


def test_calc_lst():
    gmst = 1.0
    lon = 0.5
    lst = calc_lst(gmst, lon)  # LST = GMST + lon
    assert np.isclose(lst, 1.5)


def test_convert_time():
    res = convert_time(2024, 1, 1, 12, 0, 0, 0, 0, 0, 37)
    assert isinstance(res, tuple)
    assert len(res) == 15


def test_time_utils_coverage():
    jd1, _ = calc_jd(2024, 1, 1)
    jd2, _ = calc_jd(2024, 3, 1)
    assert jd1 < jd2
    jd_neg = 2451545.0 - 100000.0
    val = calc_gmst(jd_neg)
    assert val >= 0


# STATE TRANSFORM


def test_state_transform_roundtrip():
    r = np.array([7000000.0, 0, 0])  # Define a state in ECI (Meters, m/s)
    mu = 398600.4415e9
    v_mag = np.sqrt(mu / np.linalg.norm(r))
    v = np.array([0, v_mag, 0])

    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r, v)

    assert np.isclose(a, 7000000.0, rtol=1e-6)
    assert np.isclose(ecc, 0, rtol=1e-6)
    assert np.isclose(incl, 0, rtol=1e-6)
    assert np.isclose(raan, 0, rtol=1e-6)
    assert np.isclose(argp, 0, rtol=1e-6)
    assert np.isclose(nu, 0, rtol=1e-6)
    assert np.isclose(M, 0, rtol=1e-6)
    assert np.isclose(E, 0, rtol=1e-6)
    assert np.isclose(p, 7000000.0, rtol=1e-6)
    assert np.isclose(arglat, 0, rtol=1e-6)
    assert np.isclose(truelon, 0, rtol=1e-6)
    assert np.isclose(lonper, 0, rtol=1e-6)

    r_back, v_back = kepler2eci(a, ecc, incl, raan, argp, nu)  # Round trip

    np.testing.assert_allclose(r_back, r, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(v_back, v, rtol=1e-6, atol=1e-6)


def test_anomalies_negative_mean():
    ecc = 0.5
    nu = np.pi + 0.1
    E, M = anomalies(ecc, nu)
    assert 0 <= M < 2 * np.pi
    assert 0 <= E < 2 * np.pi


# CONVERSIONS


def test_eci_ecef_roundtrip():
    reci = np.array([7000, 0, 0])
    veci = np.array([0, 7.5, 0])
    jdut1 = 2451545.0

    recef, vecef = eci2ecef(reci, veci, jdut1)
    reci_back, veci_back = ecef2eci(recef, vecef, jdut1)

    np.testing.assert_allclose(reci, reci_back, atol=1e-10)
    np.testing.assert_allclose(veci, veci_back, atol=1e-10)


def test_eci2lvlh_dcm():
    r = np.array([7000, 0, 0])
    v = np.array([0, 7.5, 0])

    expected_dcm = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

    dcm = eci2lvlh_dcm(r, v)
    np.testing.assert_allclose(dcm, expected_dcm, atol=1e-10)


def test_eci2llh():
    r_test = [6678137.0, 0, 0]
    jdut1 = 2451545.0
    lat, lon, h = eci2llh(r_test, jdut1)
    np.testing.assert_allclose(h, 300000, atol=1e-10)


def test_elements2perifocal_dcm():
    dcm = elements2perifocal_dcm(0, 0, 0)  # Identity (all zero)
    np.testing.assert_allclose(dcm, np.eye(3), atol=1e-10)

    dcm_inc = elements2perifocal_dcm(0, np.pi / 2, 0)
    np.testing.assert_allclose(np.dot(dcm_inc, dcm_inc.T), np.eye(3), atol=1e-10)

    expected_inc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(dcm_inc, expected_inc, atol=1e-10)

    dcm_raan = elements2perifocal_dcm(np.pi / 2, 0, 0)  # Case: RAAN 90 (z-rotation)
    expected_raan = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(dcm_raan, expected_raan, atol=1e-10)


def test_rotation_matrices():
    angle = np.pi / 2

    Rx = rot_x(angle)
    expected_Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(Rx, expected_Rx, atol=1e-10)

    Ry = rot_y(angle)
    expected_Ry = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    np.testing.assert_allclose(Ry, expected_Ry, atol=1e-10)

    Rz = rot_z(angle)
    expected_Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(Rz, expected_Rz, atol=1e-10)


def test_quat_dcm_roundtrip():
    q_id = np.array([0, 0, 0, 1])  # Identity
    dcm_id = quat_to_dcm(q_id)
    np.testing.assert_allclose(dcm_id, np.eye(3), atol=1e-10)
    q_out = dcm_to_quat(dcm_id)
    if np.dot(q_id, q_out) < 0:  # Quaternion double cover check
        q_out = -q_out
    np.testing.assert_allclose(q_out, q_id, atol=1e-10)

    angle = np.pi / 3
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    q_rot = np.array(
        [
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
            np.cos(angle / 2),
        ]
    )

    dcm = quat_to_dcm(q_rot)
    q_res = dcm_to_quat(dcm)

    if np.dot(q_rot, q_res) < 0:
        q_res = -q_res
    np.testing.assert_allclose(q_res, q_rot, atol=1e-10)


def test_state_to_elements_parabolic():
    mu = 398600.4415e9
    r_mag = 7000e3
    v_mag = np.sqrt(2 * mu / r_mag)
    r = np.array([r_mag, 0, 0])
    v = np.array([0, v_mag, 0])
    elements = eci2kepler(r, v)
    assert elements[0] == float("inf")


# FRAME CONVERSIONS


def test_euler_dcm_roundtrip():
    angles = np.array([0.1, 0.2, 0.3])
    seq = "321"

    dcm = euler_to_dcm(angles, seq)
    angles_out = dcm_to_euler(dcm, seq)

    np.testing.assert_allclose(angles_out, angles, atol=1e-10)


def test_euler_quat_roundtrip():
    angles = np.array([0.5, -0.2, 0.1])
    seq = "123"

    q = euler_to_quat(angles, seq)
    dcm_from_q = quat_to_dcm(q)
    dcm_from_e = euler_to_dcm(angles, seq)

    np.testing.assert_allclose(dcm_from_q, dcm_from_e, atol=1e-10)

    angles_back = quat_to_euler(q, seq)
    np.testing.assert_allclose(angles_back, angles, atol=1e-10)


def test_invalid_sequence():
    with pytest.raises(ValueError):
        euler_to_dcm([0, 0, 0], "12")
    with pytest.raises(ValueError):
        euler_to_quat([0, 0, 0], "1234")


def test_calc_jd_frac_overflow():
    jd, jdfrac = calc_jd(2000, 1, 1, hour=30)
    assert jd == 2451545.5
    assert jdfrac == pytest.approx(0.25)


def test_jd_to_datetime_edge():
    y, m, d, h, mn, s = jd_to_datetime(2451545.2, 0.0)
    assert y == 2000


def test_day_to_mdtime():
    m, d, h, mn, s = day_to_mdtime(2024, 60.5)
    assert m == 2
    assert d == 29

    m2, d2, h2, mn2, s2 = day_to_mdtime(2023, 60.5)
    assert m2 == 3
    assert d2 == 1


def test_calc_last():
    last = calc_last(2451545.0, 0.5)
    assert 0 <= last < 2 * np.pi


def test_axis_angle_to_quat_zero_axis_with_angle():
    axis = np.zeros(3)
    angle = 0.5
    q = axis_angle_to_quat(axis, angle)
    np.testing.assert_array_equal(q, np.array([0, 0, 0, 1]))


def test_axis_angle_to_quat_nonzero_axis_with_angle():
    axis = np.array([0, 0, 1.0])
    angle = 0.5
    q = axis_angle_to_quat(axis, angle)
    assert q is not None


def test_calc_gmst_negative_wrap():
    # JD that results in negative value before modulo in calc_gmst
    # ut1 = (jd - 2451545) / 36525
    # expression is roughly 67310 + 3e9 * ut1
    # For a very negative JD, ut1 is very negative.
    val = calc_gmst(-1e11)
    assert 0 <= val < 2 * np.pi


# STATE TRANSFORM
def test_eci_icrf_conversions():
    from opengnc.utils.frame_conversion import eci2icrf, icrf2eci, eci2eme2000, eme20002eci

    reci = np.array([7000.0, 0, 0])
    veci = np.array([0, 7.5, 0])
    jd = 2451545.0

    r_icrf, v_icrf = eci2icrf(reci, veci, jd)
    r_back, v_back = icrf2eci(r_icrf, v_icrf, jd)
    np.testing.assert_allclose(reci, r_back, rtol=1e-5)

    r_eme, v_eme = eci2eme2000(reci, veci)
    r_back_eme, v_back_eme = eme20002eci(r_eme, v_eme)
    np.testing.assert_allclose(reci, r_back_eme)


def test_mrp_utils():
    from opengnc.utils.mrp_utils import (
        quat_to_mrp,
        mrp_to_quat,
        mrp_to_dcm,
        get_shadow_mrp,
        check_mrp_switching,
    )

    q = np.array([0, 0, 0, 1])
    mrp = quat_to_mrp(q)
    assert np.allclose(mrp, np.zeros(3))

    q_back = mrp_to_quat(mrp)
    assert np.allclose(q_back, q)

    dcm = mrp_to_dcm(mrp)
    assert np.allclose(dcm, np.eye(3))

    shadow = get_shadow_mrp(np.array([2.0, 0, 0]))
    assert np.allclose(shadow, np.array([-0.5, 0, 0]))
    assert np.allclose(get_shadow_mrp(np.zeros(3)), np.zeros(3))

    sigma_small = np.array([0.5, 0, 0])
    res = check_mrp_switching(sigma_small)
    assert np.allclose(res, sigma_small)

    sigma_large = np.array([1.5, 0, 0])
    res_large = check_mrp_switching(sigma_large)
    assert np.allclose(res_large, get_shadow_mrp(sigma_large))


def test_jd_to_datetime_edge_negative():
    res = jd_to_datetime(-1.6, 0.0)
    assert isinstance(res, tuple)


def test_state_to_elements_edge_cases():
    from opengnc.utils.state_to_elements import rot_y as s_rot_y

    Ry = s_rot_y(0.0)
    assert np.allclose(Ry, np.eye(3))

    mu = 398600.4415e9

    # 1. Circular orbit
    r1 = np.array([0.0, 7000000.0, 0.0])
    v_mag = np.sqrt(mu / 7000000.0)
    v1 = np.array([0.0, 0.0, -v_mag])
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r1, v1)
    assert np.isclose(ecc, 0, atol=1e-6)

    # 2. Retrograde orbit
    r2 = np.array([0.0, -7000000.0, 0.0])
    v2 = np.array([-8000.0, 0.0, 0.0])
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r2, v2)
    assert np.isclose(incl, np.pi, rtol=1e-6)

    # 3. Inclined orbit (90 degrees)
    r3 = np.array([7000000.0, 0.0, 0.0])
    v3 = np.array([0.0, 0.0, 7500.0])
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r3, v3)
    assert np.isclose(incl, np.pi / 2, rtol=1e-6)

    # 4. Inclination 135 degrees
    reci, veci = kepler2eci(7000000.0, 0.1, np.pi * 0.75, 0.0, 0.0, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(incl, np.pi * 0.75, rtol=1e-6)

    # 5. RAAN 90 degrees
    reci, veci = kepler2eci(7000000.0, 0.0, 0.1, np.pi / 2, 0.0, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(raan, np.pi / 2, rtol=1e-6)

    # 6. Argument of periapsis 270 degrees
    reci, veci = kepler2eci(7000000.0, 0.1, 0.1, 0.0, np.pi * 1.5, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(argp, np.pi * 1.5, rtol=1e-6)

    # 7. Retrograde equatorial
    reci, veci = kepler2eci(7000000.0, 0.1, np.pi, 0.0, 0.0, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(incl, np.pi, rtol=1e-6)

    # 8. Hyperbolic anomaly
    reci, veci = kepler2eci(-7000000.0, 1.2, 0.1, 0.0, 0.0, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(ecc, 1.2, rtol=1e-6)

    # 9. argp e_vec[2] < 0
    reci, veci = kepler2eci(7000000.0, 0.1, 0.5, 0.0, np.pi * 1.5, 0.0)
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(reci, veci)
    assert np.isclose(argp, np.pi * 1.5, rtol=1e-6)


# KINEMATICS TESTS
def test_euler_sequences():
    from opengnc.utils.euler_utils import euler_to_dcm, dcm_to_euler

    sequences = ["321", "313", "123", "121", "232", "213"]
    angles = np.array([0.1, 0.2, 0.3])  # radians

    for seq in sequences:
        dcm = euler_to_dcm(angles, seq)
        angles_est = dcm_to_euler(dcm, seq)
        np.testing.assert_allclose(angles_est, angles, atol=1e-10)


def test_mrp_conversions_kinematics():
    from opengnc.utils.mrp_utils import quat_to_mrp, mrp_to_quat, mrp_to_dcm
    from opengnc.utils.state_conversion import quat_to_dcm

    q = np.array([0.1, 0.2, 0.3, 0.911])
    q = q / np.linalg.norm(q)

    sigma = quat_to_mrp(q)
    q_est = mrp_to_quat(sigma)
    np.testing.assert_allclose(q_est, q, atol=1e-10)

    dcm_mrp = mrp_to_dcm(sigma)
    dcm_quat = quat_to_dcm(q)
    np.testing.assert_allclose(dcm_mrp, dcm_quat, atol=1e-10)


def test_mrp_shadow_kinematics():
    from opengnc.utils.mrp_utils import mrp_to_quat, get_shadow_mrp

    sigma = np.array([0.8, 0.0, 0.0])
    sigma_shadow = get_shadow_mrp(sigma)

    q = mrp_to_quat(sigma)
    q_shadow = mrp_to_quat(sigma_shadow)

    if np.dot(q, q_shadow) < 0:
        q_shadow = -q_shadow
    np.testing.assert_allclose(q, q_shadow, atol=1e-10)


def test_crp_conversions():
    from opengnc.utils.crp_utils import quat_to_crp, crp_to_quat

    q = np.array([0.1, 0.1, 0.1, 0.98])
    q = q / np.linalg.norm(q)

    q_crp = quat_to_crp(q)
    q_est = crp_to_quat(q_crp)
    np.testing.assert_allclose(q_est, q, atol=1e-10)


def test_crp_addition():
    from opengnc.utils.crp_utils import crp_to_quat, crp_addition
    from opengnc.utils.quat_utils import quat_mult

    q1_crp = np.array([0.1, 0.0, 0.0])
    q2_crp = np.array([0.0, 0.1, 0.0])

    q_res = crp_addition(q1_crp, q2_crp)
    q1 = crp_to_quat(q1_crp)
    q2 = crp_to_quat(q2_crp)

    q_ref = quat_mult(q2, q1)
    q_est = crp_to_quat(q_res)

    if np.dot(q_est, q_ref) < 0:
        q_est = -q_est
    np.testing.assert_allclose(q_est, q_ref, atol=1e-10)


def test_cayley_klein():
    from opengnc.utils.cayley_klein_utils import quat_to_cayley_klein, cayley_klein_to_quat

    q = np.array([0.1, 0.2, 0.3, 0.911])
    q = q / np.linalg.norm(q)

    U = quat_to_cayley_klein(q)
    q_est = cayley_klein_to_quat(U)
    np.testing.assert_allclose(q_est, q, atol=1e-10)
    np.testing.assert_allclose(U @ np.conj(U).T, np.eye(2), atol=1e-10)


def test_cayley_klein_mult():
    from opengnc.utils.cayley_klein_utils import quat_to_cayley_klein, cayley_klein_mult

    U1 = quat_to_cayley_klein(np.array([0, 0, 0, 1.0]))
    U2 = quat_to_cayley_klein(np.array([0.1, 0, 0, 0.995]))
    res = cayley_klein_mult(U1, U2)
    np.testing.assert_allclose(res, U2, atol=1e-10)


def test_crp_singularities():
    from opengnc.utils.crp_utils import quat_to_crp, crp_to_dcm, crp_addition

    with pytest.raises(ValueError):
        quat_to_crp(np.array([1.0, 0.0, 0.0, 0.0]))

    dcm = crp_to_dcm(np.array([0.1, 0.2, 0.3]))
    assert dcm.shape == (3, 3)

    with pytest.raises(ValueError):
        crp_addition(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_euler_singularities():
    from opengnc.utils.euler_utils import euler_to_dcm, dcm_to_euler
    from opengnc.utils.state_conversion import rot_y

    with pytest.raises(ValueError):
        euler_to_dcm([0.1, 0.2, 0.3], "32")

    dcm_sym = np.eye(3)  # theta2 = 0 for '313'
    angles = dcm_to_euler(dcm_sym, "313")
    np.testing.assert_allclose(angles, np.zeros(3))

    dcm_asym = rot_y(np.pi / 2)  # 90 deg about Y
    angles_asym = dcm_to_euler(dcm_asym, "123")  # sequence '123'
    assert angles_asym.shape == (3,)


def test_mrp_edge_cases_kinematics():
    from opengnc.utils.mrp_utils import get_shadow_mrp, check_mrp_switching

    sigma_zero = np.zeros(3)
    res_shadow = get_shadow_mrp(sigma_zero)
    np.testing.assert_allclose(res_shadow, np.zeros(3))

    sigma_large = np.array([2.0, 0, 0])
    res_switch = check_mrp_switching(sigma_large)
    np.testing.assert_allclose(res_switch, get_shadow_mrp(sigma_large))


def test_quat_inv_singularity_kinematics():
    from opengnc.utils.quat_utils import quat_inv

    with pytest.raises(ValueError):
        quat_inv(np.zeros(4))


def test_state_conversion_singularities():
    from opengnc.utils.state_conversion import quat_to_euler, dcm_to_euler, dcm_to_quat

    with pytest.raises(ValueError):
        quat_to_euler(np.array([0, 0, 0, 1]), "12")
    with pytest.raises(ValueError):
        dcm_to_euler(np.eye(3), "12")

    R2 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    d2 = dcm_to_quat(R2)
    np.testing.assert_allclose(np.abs(d2), [1, 0, 0, 0], atol=1e-7)

    R3 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    d3 = dcm_to_quat(R3)
    np.testing.assert_allclose(np.abs(d3), [0, 1, 0, 0], atol=1e-7)

    R4 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    d4 = dcm_to_quat(R4)
    np.testing.assert_allclose(np.abs(d4), [0, 0, 1, 0], atol=1e-7)


def test_mrp_utils_coverage():
    q = np.array([0, 0, 1, -1.0])
    m = quat_to_mrp(q)
    assert m[2] > 1e11


def test_euler_utils_coverage():
    with pytest.raises(ValueError, match="Invalid axis"):
        euler_to_dcm([0, 0, 0], sequence="421")
