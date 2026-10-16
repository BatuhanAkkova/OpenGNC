import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.spatial.transform import Rotation as R
from opengnc.attitude_determination.triad import triad
from opengnc.attitude_determination.quest import quest
from opengnc.attitude_determination.davenport_q import davenport_q
from opengnc.attitude_determination.foam import foam
from opengnc.attitude_determination.request import RequestFilter, request
from opengnc.utils.quat_utils import quat_to_rmat, quat_mult, quat_inv


def test_triad_identity():
    r1 = np.array([1, 0, 0])
    r2 = np.array([0, 1, 0])
    b1 = r1
    b2 = r2

    R_est = triad(b1, b2, r1, r2)
    np.testing.assert_allclose(R_est, np.eye(3), atol=1e-6)


def test_triad_90deg_z():
    rot = R.from_euler("z", 90, degrees=True)
    R_BI_true = rot.as_matrix()

    r1 = np.array([1, 0, 0])
    r2 = np.array([0, 1, 0])

    b1 = R_BI_true @ r1
    b2 = R_BI_true @ r2

    R_est = triad(b1, b2, r1, r2)

    np.testing.assert_allclose(R_est, R_BI_true, atol=1e-6)


def test_quest_simple():
    rot = R.from_euler("xyz", [30, 45, 60], degrees=True)
    R_BI_true = rot.as_matrix()
    q_true_scipy = rot.as_quat()  # [x, y, z, w]

    r_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    b_vecs = []
    for r in r_vecs:
        b = R_BI_true @ r
        b_vecs.append(b)
    b_vecs = np.array(b_vecs)

    q_est = quest(b_vecs, r_vecs)

    if np.dot(q_est, q_true_scipy) < 0:
        q_est = -q_est

    np.testing.assert_allclose(q_est, q_true_scipy, atol=1e-6)


def test_davenport_q_simple():
    rot = R.from_euler("xyz", [30, 45, 60], degrees=True)
    R_BI_true = rot.as_matrix()
    q_true_scipy = rot.as_quat()  # [x, y, z, w]

    r_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    b_vecs = np.array([R_BI_true @ r for r in r_vecs])

    q_est = davenport_q(b_vecs, r_vecs)

    if np.dot(q_est, q_true_scipy) < 0:
        q_est = -q_est

    np.testing.assert_allclose(q_est, q_true_scipy, atol=1e-6)


def test_foam_simple():
    rot = R.from_euler("xyz", [10, 20, 30], degrees=True)
    R_BI_true = rot.as_matrix()

    r_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b_vecs = np.array([R_BI_true @ r for r in r_vecs])

    R_est = foam(b_vecs, r_vecs)

    np.testing.assert_allclose(R_est, R_BI_true, atol=1e-6)


def test_request_recursive():
    rot = R.from_euler("xyz", [15, 25, 35], degrees=True)
    R_BI_true = rot.as_matrix()
    q_true = rot.as_quat()

    r1 = np.array([[1, 0, 0], [0, 1, 0]])
    r2 = np.array([[0, 0, 1], [1, 1, 1]])

    b1 = np.array([R_BI_true @ r for r in r1])
    b2 = np.array([R_BI_true @ r for r in r2])

    q_batch = davenport_q(
        np.vstack((b1, b2)), np.vstack((r1, r2)), weights=np.array([0.25, 0.25, 0.25, 0.25])
    )

    rf = RequestFilter()
    rf.update(b1, r1, weights=np.array([0.25, 0.25]))
    rf.update(b2, r2, weights=np.array([0.25, 0.25]))
    q_rec = rf.get_quaternion()

    if np.dot(q_rec, q_batch) < 0:
        q_rec = -q_rec

    np.testing.assert_allclose(q_rec, q_batch, atol=1e-6)


def test_triad_collinear_error():
    v = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        triad(v, v, v, v)


def test_quest_mismatch_error():
    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((2, 3))
    with pytest.raises(ValueError):
        quest(b_vecs, r_vecs)


def test_davenport_q_mismatch_error():
    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((2, 3))
    with pytest.raises(
        ValueError, match="Body and reference vector arrays must have the same shape."
    ):
        davenport_q(b_vecs, r_vecs)

    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((3, 3))
    weights = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="Number of weights must match number of vectors."):
        davenport_q(b_vecs, r_vecs, weights=weights)


def test_foam_mismatch_error():
    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((2, 3))
    with pytest.raises(
        ValueError, match="Body and reference vector arrays must have the same shape."
    ):
        foam(b_vecs, r_vecs)

    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((3, 3))
    weights = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="Number of weights must match number of vectors."):
        foam(b_vecs, r_vecs, weights=weights)


def test_foam_singular_B():
    r_vecs = np.array([[1, 0, 0], [0, 1, 0]])
    b_vecs = np.array([[1, 0, 0], [0, 1, 0]])
    R_est = foam(b_vecs, r_vecs)
    assert R_est.shape == (3, 3)


def test_request_additional():
    K_init = np.eye(4)
    rf = RequestFilter(initial_K=K_init)
    np.testing.assert_array_equal(rf.K, K_init)

    r_vecs = np.array([[1, 0, 0], [0, 1, 0]])
    b_vecs = np.array([[1, 0, 0], [0, 1, 0]])
    rf.update(b_vecs, r_vecs, weights=None)

    q = request(b_vecs, r_vecs)
    assert q.shape == (4,)


def test_triad_ref_collinear_error():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="Reference vectors are collinear or nearly collinear."):
        triad(v1, v2, v1, v1)


def test_quest_with_weights():
    rot = R.from_euler("xyz", [45, 45, 45], degrees=True)
    R_BI_true = rot.as_matrix()
    q_true_scipy = rot.as_quat()

    r_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b_vecs = np.array([R_BI_true @ r for r in r_vecs])
    weights = np.array([0.5, 0.3, 0.2])

    q_est = quest(b_vecs, r_vecs, weights=weights)
    if np.dot(q_est, q_true_scipy) < 0:
        q_est = -q_est
    np.testing.assert_allclose(q_est, q_true_scipy, atol=1e-6)


def test_quest_weight_mismatch():
    b = np.eye(3)
    r = np.eye(3)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Weight vector dimension mismatch."):
        quest(b, r, weights=w)


def test_request_k_setter():
    rf = RequestFilter()
    new_k = np.eye(4) * 2
    rf.K = new_k
    assert np.allclose(rf.k, new_k)
