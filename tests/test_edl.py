import unittest
import numpy as np
from opengnc.edl import (
    ballistic_entry_dynamics,
    lifting_entry_dynamics,
    sutton_grave_heating,
    calculate_g_load,
    aerocapture_guidance,
    hazard_avoidance,
)
from opengnc.environment.density import Exponential
from unittest.mock import patch, MagicMock


class TestEDL(unittest.TestCase):
    def test_ballistic_entry(self):
        r = np.array([6491000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        state = np.concatenate([r, v])

        cd = 2.0
        area = 1.0  # m^2
        mass = 100.0  # kg

        deriv = ballistic_entry_dynamics(0.0, state, cd, area, mass)

        self.assertEqual(len(deriv), 6)
        np.testing.assert_allclose(deriv[:3], v)

        v_unit = v / np.linalg.norm(v)
        r_unit = r / np.linalg.norm(r)

        self.assertLess(deriv[3], 0)  # Gravity
        self.assertLess(deriv[4], 0)  # Drag

    def test_lifting_entry(self):
        r = np.array([6491000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        state = np.concatenate([r, v])

        cl = 0.5
        cd = 1.0
        bank = 0.0
        area = 2.0
        mass = 200.0

        deriv_lift = lifting_entry_dynamics(0.0, state, cl, cd, bank, area, mass)
        deriv_ballistic = ballistic_entry_dynamics(0.0, state, cd, area, mass)

        self.assertGreater(deriv_lift[3], deriv_ballistic[3])

    def test_heating(self):
        rho = 1e-4
        v = 7500.0
        rn = 1.0
        q = sutton_grave_heating(rho, v, rn)

        self.assertGreater(q, 100000)

    def test_g_load(self):
        acc = np.array([0.0, 0.0, 9.80665 * 5])
        g = calculate_g_load(acc)
        self.assertAlmostEqual(g, 5.0)

    def test_aerocapture_guidance_no_lift(self):
        res = aerocapture_guidance(None, 0.0, 0.0, 0.0, 0.0, {}, None, cl=0.0)
        self.assertEqual(res, 0.0)

    def test_hazard_avoidance(self):
        r = np.array([0, 0, 0])
        v = np.array([1, 1, 1])
        hazards = [np.array([100, 100, 100])]  # far away

        res = hazard_avoidance(r, v, hazards, safety_margin=50.0)
        np.testing.assert_allclose(res, np.zeros(3))

        hazards_near = [np.array([10, 0, 0])]
        res_near = hazard_avoidance(r, v, hazards_near, safety_margin=50.0)
        self.assertTrue(np.linalg.norm(res_near) > 0)

    def test_aerocapture_guidance_bisection(self):
        r = np.array([6491000.0, 0.0, 0.0])
        v_ok = np.array([0.0, 7500.0, 0.0])
        state_ok = np.concatenate([r, v_ok])

        planet_params = {"mu": 3.986e14, "r_planet": 6371000.0}
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8.5)

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            count = 0

            def side_effect(*args, **kwargs):
                nonlocal count
                count += 1
                sol = MagicMock()
                if count == 1:
                    sol.y = np.array([[6500000, 0, 0, 0, 15000, 0]]).T
                elif count == 2:
                    sol.y = np.array([[6400000, 0, 0, 0, 7000, 0]]).T
                else:
                    sol.y = np.array([[6492000, 0, 0, 0, 7400, 0]]).T
                return sol

            mock_solve.side_effect = side_effect

            res = aerocapture_guidance(
                state_ok, 121000.0, 1.0, 1.0, 100.0, planet_params, rho_model, cl=0.5
            )
            self.assertTrue(isinstance(res, float))

    def test_edl_advanced(self):
        r = np.array([0.0, 0.0, 10.0])
        v = np.array([0.0, 0.0, -1.0])
        hazards = [np.array([0.0, 0.0, 0.0])]
        dv = hazard_avoidance(r, v, hazards, safety_margin=20.0)
        self.assertGreater(np.linalg.norm(dv), 0)

        r_planet = 6371000.0
        state = np.array([r_planet + 115000.0, 0.0, 0.0, -100.0, 10800.0, 0.0])
        target_apoapsis = 500000.0
        cd = 1.0
        cl = 0.3
        area = 5.0
        mass = 1000.0
        planet_params = {"mu": 3.986e14, "r_planet": r_planet}
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8.5)

        bank = aerocapture_guidance(
            state, target_apoapsis, cd, area, mass, planet_params, rho_model, cl=cl
        )
        self.assertTrue(0.0 <= bank <= np.pi)


if __name__ == "__main__":
    unittest.main()
