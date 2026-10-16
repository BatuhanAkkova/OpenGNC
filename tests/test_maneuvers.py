import unittest
import numpy as np
from opengnc.guidance.maneuvers import (
    delta_v_budget,
    raan_change,
    optimal_combined_maneuver,
    hohmann_transfer,
)


class TestManeuversExtended(unittest.TestCase):
    def test_delta_v_budget(self):
        initial_mass = 1000.0  # kg
        dv = 0.5  # km/s
        isp = 300.0  # s

        m_prop = delta_v_budget(initial_mass, dv, isp)

        self.assertAlmostEqual(m_prop, 156.3, delta=1.0)
        self.assertGreater(m_prop, 0)
        self.assertLess(m_prop, initial_mass)

    def test_raan_change(self):
        v = 7.5  # km/s
        i = np.radians(28.5)
        delta_raan = np.radians(1.0)

        dv = raan_change(v, i, delta_raan)

        self.assertAlmostEqual(dv, 0.0624, delta=0.01)

    def test_optimal_combined_maneuver(self):
        r1 = 7000.0
        r2 = 42164.0
        delta_i = np.radians(28.5)
        mu = 398600.4418

        dv_opt, di1, di2 = optimal_combined_maneuver(r1, r2, delta_i, mu)

        self.assertAlmostEqual(di1 + di2, delta_i)

        dv1_h, dv2_h, _ = hohmann_transfer(r1, r2, mu)
        v_c2 = np.sqrt(mu / r2)
        dv_pc = 2 * v_c2 * np.sin(delta_i / 2.0)
        total_non_opt = dv1_h + dv2_h + dv_pc

        self.assertLess(dv_opt, total_non_opt)

        from opengnc.guidance.maneuvers import combined_plane_change

        a_trans = (r1 + r2) / 2.0
        v_trans_a = np.sqrt(mu * (2 / r2 - 1 / a_trans))
        dv2_combined = combined_plane_change(v_trans_a, v_c2, delta_i)
        total_combined_at_r2 = dv1_h + dv2_combined

        self.assertLessEqual(dv_opt, total_combined_at_r2 + 1e-6)


if __name__ == "__main__":
    unittest.main()
