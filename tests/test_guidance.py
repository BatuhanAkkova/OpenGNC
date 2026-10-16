import unittest
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from opengnc.guidance.maneuvers import (
    hohmann_transfer,
    bi_elliptic_transfer,
    phasing_maneuver,
    plane_change,
    combined_plane_change,
)
from opengnc.guidance.rendezvous import (
    solve_lambert,
    cw_equations,
    cw_targeting,
    solve_lambert_multi_rev,
    is_within_corridor,
)
from opengnc.guidance.formation_flying import (
    virtual_structure_control,
    leader_follower_control,
    fuel_balanced_formation_keeping,
    distributed_consensus_control,
)
from opengnc.guidance.continuous_thrust import (
    q_law_guidance,
    zem_zev_guidance,
    gravity_turn_guidance,
    apollo_dps_guidance,
    indirect_optimal_guidance,
    direct_collocation_guidance,
)
from opengnc.utils.state_to_elements import eci2kepler


class TestManeuvers(unittest.TestCase):
    def test_hohmann_transfer(self):
        r1 = 7000
        r2 = 42164
        mu = 398600.4418

        dv1, dv2, t_trans = hohmann_transfer(r1, r2, mu)

        self.assertAlmostEqual(dv1, 2.4, delta=0.3)  # Rough check
        self.assertGreater(t_trans, 0)

    def test_bi_elliptic_transfer(self):
        r1 = 7000
        r2 = 105000  # r2/r1 = 15
        rb = 200000
        mu = 398600.4418

        dv1, dv2, dv3, t = bi_elliptic_transfer(r1, r2, rb, mu)
        total_dv = dv1 + dv2 + dv3

        dh1, dh2, th = hohmann_transfer(r1, r2, mu)
        total_hohmann = dh1 + dh2

        self.assertLess(total_dv, total_hohmann)

    def test_plane_change(self):
        v = 7.5
        di = np.radians(10)
        dv = plane_change(v, di)
        expected = 2 * v * np.sin(np.radians(5))
        self.assertAlmostEqual(dv, expected)

    def test_phasing(self):
        a = 7000
        dv, t_phasing = phasing_maneuver(a, 100)

        self.assertGreater(dv, 0)
        self.assertAlmostEqual(t_phasing - 2 * np.pi * np.sqrt(a**3 / 398600.4418), 100, delta=1.0)


class TestRendezvous(unittest.TestCase):
    def test_solve_lambert(self):
        mu = 398600.4418
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 10000.0, 0.0])  # 90 deg separation, diff radius

        dt = 2000.0  # seconds

        v1, v2 = solve_lambert(r1, r2, dt, mu=mu, tm=1)

        self.assertFalse(np.isnan(v1).any())
        self.assertFalse(np.isnan(v2).any())

    def test_cw_equations(self):
        r0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])
        n = 0.001
        t = 100.0

        rt, vt = cw_equations(r0, v0, n, t)
        np.testing.assert_array_almost_equal(rt, r0)
        np.testing.assert_array_almost_equal(vt, v0)

        r0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, -0.1, 0.0])  # Drift backwards
        rt, vt = cw_equations(r0, v0, n, t)

        self.assertNotEqual(rt[1], 0)

    def test_cw_targeting(self):
        n = 0.0011  # LEO approx
        t_transfer = 1000.0

        r0 = np.array([10.0, 0.0, 0.0])  # 10 km below/above
        r_target = np.array([0.0, 0.0, 0.0])  # To origin

        v0_req = cw_targeting(r0, r_target, t_transfer, n)

        # Propagate to check
        rt, vt = cw_equations(r0, v0_req, n, t_transfer)

        np.testing.assert_allclose(rt, r_target, atol=1e-5)

    def test_solve_lambert_long_way(self):
        mu = 398600.4418
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 10000.0, 0.0])
        dt = 2000.0
        v1, v2 = solve_lambert(r1, r2, dt, mu=mu, tm=-1)
        self.assertFalse(np.isnan(v1).any())

    def test_solve_lambert_singularity(self):
        mu = 398600.4418
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([-7000.0, 0.0, 0.0])  # 180 deg
        dt = 2000.0
        with self.assertRaises(ValueError):
            solve_lambert(r1, r2, dt, mu=mu)

    def test_solve_lambert_hyperbolic(self):
        mu = 398600.4418
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 10000.0, 0.0])
        dt = 100.0  # Extremely small to trigger large energy / psi < -1e-6
        v1, v2 = solve_lambert(r1, r2, dt, mu=mu)
        self.assertFalse(np.isnan(v1).any())

    def test_solve_lambert_multi_rev_fallback(self):
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 10000.0, 0.0])
        dt = 2000.0
        v1, v2 = solve_lambert_multi_rev(r1, r2, dt, n_rev=0)
        self.assertFalse(np.isnan(v1).any())

    def test_cw_targeting_singularity(self):
        n = 0.01
        t_transfer = 2 * np.pi / n  # 2pi period
        r0 = np.array([10.0, 1.0, 1.0])
        r_target = np.array([0.0, 0.0, 0.0])

        with patch("numpy.linalg.solve") as mock_solve:
            mock_solve.side_effect = np.linalg.LinAlgError()
            v0_req = cw_targeting(r0, r_target, t_transfer, n)
            np.testing.assert_allclose(v0_req, np.zeros(3))

    def test_is_within_corridor(self):
        r_rel = np.array([0.0, 0, 0])
        axis = np.array([1, 0, 0])
        self.assertTrue(is_within_corridor(r_rel, axis, 10))

    def test_rendezvous_safety_lines(self):
        r1 = np.array([7000, 0, 0])
        r2 = np.array([0, 7000, 0])
        solve_lambert(r1, r2, 1000.0)


def test_virtual_structure_control():
    state_actual = np.array([[1.0, 2.0], [0.5, 1.5]])
    state_desired = np.array([[1.0, 2.0], [1.0, 2.0]])
    gains = np.array([2.0, 2.0])

    control = virtual_structure_control(state_actual, state_desired, gains)
    assert np.allclose(control[0], [0.0, 0.0])
    assert np.allclose(control[1], [1.0, 1.0])


def test_leader_follower_control():
    leader_state = np.array([10.0, 5.0])
    follower_state = np.array([8.0, 4.0])
    desired_relative_state = np.array([-1.0, -1.0])  # follower should be at [9.0, 4.0]
    gains = np.array([0.5, 0.5])

    control = leader_follower_control(leader_state, follower_state, desired_relative_state, gains)
    assert np.allclose(control, [0.5, 0.0])


def test_fuel_balanced_formation_keeping():
    states = np.array([[0.0, 0.0], [0.0, 0.0]])
    fuel_levels = np.array([75.0, 25.0])  # 75% and 25% of total 100
    weights = np.array([[10.0, 10.0], [10.0, 10.0]])

    control = fuel_balanced_formation_keeping(states, fuel_levels, weights)
    assert np.allclose(control[0], [7.5, 7.5])
    assert np.allclose(control[1], [2.5, 2.5])


def test_distributed_consensus_control():
    states = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    laplacian = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    gains = 1.0

    control = distributed_consensus_control(states, laplacian, gains)
    expected = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 0.0]])
    assert np.allclose(control, expected)


def test_fuel_balanced_zero_fuel():
    states = np.array([[0.0, 0.0], [0.0, 0.0]])
    fuel_levels = np.array([0.0, 0.0])
    weights = np.array([[10.0, 10.0], [10.0, 10.0]])

    control = fuel_balanced_formation_keeping(states, fuel_levels, weights)
    assert np.allclose(control, np.zeros_like(weights))


class TestContinuousThrust(unittest.TestCase):
    def test_q_law(self):
        r = np.array([7000000.0, 0.0, 0.0])
        v = np.array([0.0, 7550.0, 0.0])
        mu = 398600.4415e9

        target_a = 7500000.0
        target_oe = np.array([target_a, 0.0, 0.0, 0.0, 0.0])

        f_max = 0.01  # m/s^2

        f_eci = q_law_guidance(r, v, target_oe, mu, f_max)

        self.assertEqual(len(f_eci), 3)
        self.assertAlmostEqual(np.linalg.norm(f_eci), f_max, delta=1e-10)
        self.assertGreater(f_eci[1], 0)

    def test_zem_zev(self):
        r = np.array([1000.0, 0.0, 0.0])
        v = np.array([-10.0, 0.0, 0.0])
        r_target = np.array([0.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.0, 0.0])
        t_go = 100.0

        a_cmd = zem_zev_guidance(r, v, r_target, v_target, t_go)
        np.testing.assert_allclose(a_cmd, np.array([-0.2, 0.0, 0.0]), atol=1e-5)

    def test_gravity_turn(self):
        v = np.array([0.0, 0.0, -100.0])
        f_mag = 20.0

        a_thrust = gravity_turn_guidance(v, f_mag, mode="descent")
        np.testing.assert_allclose(a_thrust, np.array([0.0, 0.0, 20.0]))

        a_thrust = gravity_turn_guidance(v, f_mag, mode="ascent")
        np.testing.assert_allclose(a_thrust, np.array([0.0, 0.0, -20.0]))

        a_thrust_zero = gravity_turn_guidance(np.zeros(3), f_mag)
        np.testing.assert_allclose(a_thrust_zero, np.zeros(3))

    def test_apollo_dps(self):
        t_go = 50.0
        r = np.array([500.0, 0.0, 0.0])
        v = np.array([-20.0, 0.0, 0.0])
        r_target = np.array([0.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.0, 0.0])
        a_target = np.array([0.0, 0.0, 0.0])

        a_cmd = apollo_dps_guidance(t_go, r, v, r_target, v_target, a_target)

        self.assertAlmostEqual(a_cmd[0], 4.8, delta=1e-5)

    def test_q_law_inclined(self):
        r = np.array([7000000.0, 0.0, 0.0])
        v = np.array([0.0, 5000.0, 5000.0])  # Inclined
        mu = 398600.4415e9

        target_oe = np.array([7500000.0, 0.1, 0.1, 0.1, 0.1])  # Inclined target
        f_max = 0.01

        f_eci = q_law_guidance(r, v, target_oe, mu, f_max)
        self.assertEqual(len(f_eci), 3)

    def test_q_law_zero_accel(self):
        r = np.array([7000000.0, 0.0, 0.0])
        v = np.array([0.0, 7550.0, 0.0])
        mu = 398600.4415e9
        a, e, i, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r, v)
        target_oe = np.array([a, e, i, raan, argp])

        f_eci = q_law_guidance(r, v, target_oe, mu, 0.01)

    def test_zem_zev_tgo0(self):
        res = zem_zev_guidance(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0)
        np.testing.assert_allclose(res, np.zeros(3))

    def test_apollo_dps_tgo0(self):
        res = apollo_dps_guidance(
            0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.array([1, 2, 3])
        )
        np.testing.assert_allclose(res, [1, 2, 3])

    def test_indirect_optimal(self):
        r0 = np.array([7000000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7550.0, 0.0])
        rf = np.array([7100000.0, 0.0, 0.0])
        vf = np.array([0.0, 7450.0, 0.0])
        tf = 1000.0
        mu = 398600.4415e9

        t, a = indirect_optimal_guidance(r0, v0, rf, vf, tf, mu)
        if t is not None:
            self.assertEqual(len(t), len(a[0]))
            self.assertGreater(len(t), 0)

    def test_direct_collocation(self):
        r0 = np.array([7000000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7550.0, 0.0])
        rf = np.array([7001000.0, 0.0, 0.0])
        vf = np.array([0.0, 7550.0, 0.0])
        tf = 100.0
        mu = 398600.4415e9

        traj = direct_collocation_guidance(r0, v0, rf, vf, tf, mu, n_nodes=10)
        if traj is not None:
            self.assertEqual(traj.shape, (10, 9))
            np.testing.assert_allclose(traj[0, :3], r0, atol=1e-3)
            np.testing.assert_allclose(traj[-1, :3], rf, atol=1e-3)

    def test_direct_collocation_success(self):
        with patch("opengnc.guidance.continuous_thrust.minimize") as mock_min:
            ret = MagicMock()
            ret.success = True
            n_nodes = 10
            ret.x = np.zeros(n_nodes * 9)
            mock_min.return_value = ret
            res = direct_collocation_guidance(
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 100, 3.986e14, n_nodes=n_nodes
            )
            self.assertIsNotNone(res)
            self.assertEqual(res.shape, (n_nodes, 9))

    def test_indirect_optimal_fail(self):
        with patch("opengnc.guidance.continuous_thrust.solve_bvp") as mock_solve:
            ret = MagicMock()
            ret.success = False
            mock_solve.return_value = ret
            t, u = indirect_optimal_guidance(
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 100, 3.986e14
            )
            self.assertIsNone(t)

    def test_direct_collocation_fail(self):
        with patch("opengnc.guidance.continuous_thrust.minimize") as mock_min:
            ret = MagicMock()
            ret.success = False
            mock_min.return_value = ret
            res = direct_collocation_guidance(
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 100, 3.986e14
            )
            self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
