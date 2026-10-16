import numpy as np
import pytest
from opengnc.guidance import (
    nadir_pointing_reference,
    sun_pointing_reference,
    target_tracking_reference,
    attitude_blending,
    eigenaxis_slew_path_planning,
)
from opengnc.utils.quat_utils import quat_norm, quat_rot, quat_conj


def test_nadir_pointing():
    pos = np.array([7000e3, 0.0, 0.0])  # 7000 km on X
    vel = np.array([0.0, 7500.0, 0.0])  # 7.5 km/s on Y

    q = nadir_pointing_reference(pos, vel)

    assert quat_norm(q) == pytest.approx(1.0)

    # In Nadir frame:
    # Body Z should align with -pos (towards Earth)
    # Body Y should align with -orb_normal
    # orb_normal = pos x vel = [0, 0, 7000e3 * 7500] (Positive Z)
    # So Body Y should align with -Z_eci

    body_z = quat_rot(quat_conj(q), [0, 0, 1.0])
    assert body_z[0] == pytest.approx(-1.0)  # Aligned with -X_eci

    body_y = quat_rot(quat_conj(q), [0, 1.0, 0.0])
    assert body_y[2] == pytest.approx(-1.0)  # Aligned with -Z_eci


def test_sun_pointing():
    sun_vec = np.array([0.0, 1.0, 0.0])  # Sun on Y_eci
    # Primary axis is Body X
    q = sun_pointing_reference(sun_vec)

    body_x = quat_rot(q, [1.0, 0.0, 0.0])
    assert body_x[0] == pytest.approx(0.0, abs=1e-7)
    assert body_x[1] == pytest.approx(1.0)


def test_attitude_blending():
    q1 = np.array([0, 0, 0, 1.0])  # Identity
    # 90 deg rotation about Z
    q2 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    # Halfway (45 deg)
    q_mid = attitude_blending(q1, q2, 0.5)

    expected_mid = np.array([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    np.testing.assert_allclose(q_mid, expected_mid, atol=1e-7)


def test_target_tracking():
    pos = np.array([0, 0, 0])
    target = np.array([100.0, 100.0, 100.0])

    q = target_tracking_reference(pos, target)

    # Body Z should point towards target
    body_z = quat_rot(quat_conj(q), [0, 0, 1.0])
    expected_z = target / np.linalg.norm(target)
    np.testing.assert_allclose(body_z, expected_z, atol=1e-7)


def test_sun_pointing_aligned():
    # Dot > 0.999999
    q1 = sun_pointing_reference(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(q1, [0.0, 0.0, 0.0, 1.0])

    # Dot < -0.999999
    q2 = sun_pointing_reference(np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(q2, [0.0, 1.0, 0.0, 0.0])


def test_target_tracking_aligned():
    # LOS aligned with Z-axis
    pos = np.array([0, 0, 0])
    target = np.array([0, 0, 10.0])
    q = target_tracking_reference(pos, target)
    assert quat_norm(q) == pytest.approx(1.0)


def test_eigenaxis_slew():
    q1 = np.array([0, 0, 0, 1.0])
    q2 = np.array([0, 0, 1.0, 0.0])  # 180 deg rotation
    path = eigenaxis_slew_path_planning(q1, q2, np.array([0, 0.5, 1.0]))
    assert len(path) == 3


def test_attitude_blending_edge_cases():
    q1 = np.array([0, 0, 0, 1.0])

    # Antipodal
    q_anti = attitude_blending(q1, -q1, 0.5)
    np.testing.assert_allclose(quat_norm(q_anti), 1.0)

    # Close
    angle = 0.01
    q_close = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    q_blend = attitude_blending(q1, q_close, 0.5)
    np.testing.assert_allclose(quat_norm(q_blend), 1.0)


def test_rmat_to_quat():
    from opengnc.guidance.attitude_guidance import _rmat_to_quat

    # tr > 0
    R1 = np.eye(3)
    q1 = _rmat_to_quat(R1)
    np.testing.assert_allclose(q1, [0, 0, 0, 1.0])

    # R00 max, tr <= 0
    # 180 deg about X
    R2 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    q2 = _rmat_to_quat(R2)
    # can be [1,0,0,0] or [-1,0,0,0]
    # abs check
    np.testing.assert_allclose(np.abs(q2), [1, 0, 0, 0], atol=1e-7)

    # R11 max, tr <= 0
    # 180 deg about Y
    R3 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    q3 = _rmat_to_quat(R3)
    np.testing.assert_allclose(np.abs(q3), [0, 1, 0, 0], atol=1e-7)

    # R22 max, tr <= 0
    # 180 deg about Z
    R4 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    q4 = _rmat_to_quat(R4)
    np.testing.assert_allclose(np.abs(q4), [0, 0, 1, 0], atol=1e-7)
