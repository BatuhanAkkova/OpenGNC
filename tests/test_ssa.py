import unittest
import numpy as np
from opengnc.ssa.collision_avoidance import plan_collision_avoidance_maneuver


class TestCollisionAvoidance(unittest.TestCase):
    def test_cam_optimization(self):
        # Setup a conjunction scenario at TCA
        r_sat = np.array([7000e3, 0, 0])
        v_sat = np.array([0, 7500.0, 0])
        cov_sat = np.eye(3) * 100.0  # 10m sigma

        # Debris is very close (direct collision)
        r_deb = r_sat + np.array([5.0, 5.0, 5.0])
        v_deb = np.array([0, 0, 7500.0])
        cov_deb = np.eye(3) * 100.0

        hbr = 10.0  # 10m radius
        t_man = 3600.0  # 1 hour before TCA

        # Initial Pc should be high
        from opengnc.ssa.conjunction import compute_pc_chan

        pc_initial = compute_pc_chan(r_sat, v_sat, cov_sat, r_deb, v_deb, cov_deb, hbr)
        print(f"Initial Pc: {pc_initial:.4e}")

        # Optimized CAM
        dv, pc_final = plan_collision_avoidance_maneuver(
            r_sat, v_sat, cov_sat, r_deb, v_deb, cov_deb, hbr, t_man, pc_limit=1e-6
        )

        print(f"DV Vector (ECI): {dv}")
        print(f"DV Magnitude: {np.linalg.norm(dv):.4f} m/s")
        print(f"Final Pc: {pc_final:.4e}")

        self.assertLessEqual(pc_final, 1.1e-6)  # Allow some tolerance
        self.assertGreater(np.linalg.norm(dv), 0.0)


if __name__ == "__main__":
    unittest.main()
