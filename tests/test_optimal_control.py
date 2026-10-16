import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from opengnc.optimal_control import (
    LinearMPC,
    NonlinearMPC,
    SlidingModeController,
    CasadiNMPC,
    FeedbackLinearization,
    FiniteHorizonLQR,
    HInfinityController,
    H2Controller,
    LQR,
    LQE,
    LQG,
    ModelReferenceAdaptiveControl,
    INDIController,
    INDIOuterLoopPD,
    GeometricController,
    PassivityBasedController,
    BacksteppingController,
)

try:
    import casadi as ca
except ImportError:
    ca = None


class TestLQR(unittest.TestCase):
    def test_double_integrator(self):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.eye(1)

        lqr = LQR(A, B, Q, R)
        K = lqr.compute_gain()

        self.assertEqual(K.shape, (1, 2))

        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)

        for eig in eigenvalues:
            self.assertLess(eig.real, 0)

    def test_lqe_scalar(self):
        A = [[-1]]
        G = [[1]]
        C = [[1]]
        Q = [[1]]
        R = [[1]]

        lqe = LQE(A, G, C, Q, R)
        P = lqe.solve()
        L = lqe.compute_gain()

        expected_P = np.sqrt(2) - 1
        self.assertAlmostEqual(P[0, 0], expected_P, places=5)
        self.assertAlmostEqual(L[0, 0], expected_P, places=5)

    def test_sliding_mode(self):
        surface_func = lambda x, t: x[0]
        K = 1.0
        smc = SlidingModeController(surface_func, K, chattering_reduction=False)

        x_pos = np.array([2.0])
        u_pos = smc.compute_control(x_pos)
        self.assertEqual(u_pos, -1.0)

        x_neg = np.array([-2.0])
        u_neg = smc.compute_control(x_neg)
        self.assertEqual(u_neg, 1.0)

        smc_sat = SlidingModeController(
            surface_func, K, chattering_reduction=True, boundary_layer=1.0
        )
        x_iny = np.array([0.5])
        u_sat = smc_sat.compute_control(x_iny)
        self.assertEqual(u_sat, -0.5)

    def test_mpc(self):
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt]])
        Q = np.diag([1.0, 0.1])
        R = np.eye(1) * 0.01

        mpc = LinearMPC(
            A, B, Q, R, horizon=10, u_min=[-1.0], u_max=np.array([1.0]), x_max=np.array([1.5, 10.0])
        )

        x0 = np.array([1.0, 0.0])

        U = mpc.solve(x0)

        self.assertEqual(U.shape, (10, 1))

        self.assertLess(U[0, 0], 0)

        self.assertTrue(np.all(U >= -1.0))
        self.assertTrue(np.all(U <= 1.0))

        x_curr = x0.copy()
        for i in range(10):
            x_curr = A @ x_curr + B @ U[i]
            self.assertLessEqual(x_curr[0], 1.5001)

    def test_nonlinear_mpc(self):
        dt = 0.1

        def dynamics(x, u):
            x1, x2 = x
            x1_next = x1 + x2 * dt
            x2_next = x2 + (-np.sin(x1) + u[0]) * dt
            return np.array([x1_next, x2_next])

        def cost(x, u):
            return (x[0] ** 2 + x[1] ** 2) + 0.1 * u[0] ** 2

        def terminal_cost(x):
            return 10.0 * (x[0] ** 2 + x[1] ** 2)

        nmpc = NonlinearMPC(
            dynamics_func=dynamics,
            cost_func=cost,
            terminal_cost_func=terminal_cost,
            horizon=10,
            nx=2,
            nu=1,
            u_min=[-2.0],
            u_max=[2.0],
        )

        x0 = np.array([np.pi / 2, 0.0])

        U = nmpc.solve(x0)

        self.assertEqual(U.shape, (10, 1))

        self.assertTrue(np.all(U >= -2.001))
        self.assertTrue(np.all(U <= 2.001))

    def test_mpc_bounds_none(self):
        A = np.eye(2)
        B = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2)
        mpc = LinearMPC(A, B, Q, R, horizon=5)
        u = mpc.solve([1, 1])
        self.assertEqual(u.shape, (5, 2))

    def test_mpc_dimension_mismatch(self):
        A = np.eye(2)
        B = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2)
        with self.assertRaises(ValueError):
            mpc = LinearMPC(A, B, Q, R, horizon=5, u_min=[1])
            mpc.solve([1, 1])
        with self.assertRaises(ValueError):
            mpc = LinearMPC(A, B, Q, R, horizon=5, x_min=[1, 2, 3])
            mpc.solve([1, 1])
        with self.assertRaises(ValueError):
            mpc = LinearMPC(A, B, Q, R, horizon=5, x_max=[1, 2, 3])
            mpc.solve([1, 1])

    def test_mpc_x_min(self):
        A = np.eye(2)
        B = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2)
        mpc = LinearMPC(A, B, Q, R, horizon=5, x_min=[-1.0, -1.0])
        u = mpc.solve([1, 1])
        self.assertEqual(u.shape, (5, 2))

    def test_nonlinear_mpc_constraints(self):
        def f(x, u):
            return x + u

        def L(x, u):
            return x[0] ** 2 + u[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = NonlinearMPC(f, L, V, horizon=5, nx=2, nu=2, x_min=[-1.0, -1.0], x_max=[1.0, 1.0])
        u = nmpc.solve([0, 0])
        self.assertEqual(u.shape, (5, 2))

    def test_nonlinear_mpc_bounds_none(self):
        def f(x, u):
            return x + u

        def L(x, u):
            return x[0] ** 2 + u[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = NonlinearMPC(f, L, V, horizon=5, nx=2, nu=2)
        u = nmpc.solve([0, 0])
        self.assertEqual(u.shape, (5, 2))

    def test_nonlinear_mpc_mismatch(self):
        def f(x, u):
            return x + u

        def L(x, u):
            return x[0] ** 2 + u[0] ** 2

        def V(x):
            return x[0] ** 2

        with self.assertRaises(ValueError):
            nmpc = NonlinearMPC(f, L, V, horizon=5, nx=2, nu=2, u_min=[1])
            nmpc.solve([0, 0])

    def test_mpc_infeasible(self):
        A = np.eye(2)
        B = np.eye(2)
        Q = np.eye(2)
        mpc = LinearMPC(A, B, Q, np.eye(2), horizon=5, x_min=[10.0, 10.0], x_max=[-10.0, -10.0])
        u = mpc.solve([0.0, 0.0])
        self.assertEqual(u.shape, (5, 2))

        def f(x, u):
            return x + u

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = NonlinearMPC(
            f, L, V, horizon=5, nx=2, nu=2, x_min=[10.0, 10.0], x_max=[-10.0, -10.0]
        )
        u_nl = nmpc.solve([0.0, 0.0])
        self.assertEqual(u_nl.shape, (5, 2))

    def test_mpc_coverage(self):
        A = np.zeros((1, 1))
        B = np.ones((1, 1))

        with self.assertRaises(ValueError):
            LinearMPC(A, B, np.eye(1), np.eye(1), horizon=5, u_min=np.array([0, 0]))
        with self.assertRaises(ValueError):
            LinearMPC(A, B, np.eye(1), np.eye(1), horizon=5, u_max=np.array([0, 0]))

        mpc = LinearMPC(A, B, np.eye(1), np.eye(1), horizon=5)
        mpc.solve(np.array([1.0]))

        def d(x, u):
            return x + u

        def c(x, u):
            return float(np.sum(x**2 + u**2))

        def tc(x):
            return float(np.sum(x**2))

        with self.assertRaises(ValueError):
            NonlinearMPC(d, c, tc, 5, 1, 2, u_min=np.array([0]))
        with self.assertRaises(ValueError):
            NonlinearMPC(d, c, tc, 5, 1, 2, u_max=np.array([0, 0, 0]))
        with self.assertRaises(ValueError):
            NonlinearMPC(d, c, tc, 5, 2, 1, x_min=np.array([0]))
        with self.assertRaises(ValueError):
            NonlinearMPC(d, c, tc, 5, 2, 1, x_max=np.array([0, 0, 0]))

    def test_feedback_linearization(self):
        f_func = lambda x: x**2
        g_func = lambda x: 1.0

        fl_controller = FeedbackLinearization(f_func, g_func)

        x = np.array([2.0])
        v = -x
        u = fl_controller.compute_control(x, v)

        self.assertEqual(u, -6.0)

    def test_feedback_linearization_matrix(self):
        f_func = lambda x: np.zeros(2)
        g_func = lambda x: np.eye(2)
        fl_controller = FeedbackLinearization(f_func, g_func)
        x = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        u = fl_controller.compute_control(x, v)
        np.testing.assert_allclose(u, [3.0, 4.0])


class TestLQG(unittest.TestCase):
    def test_lqg_stability(self):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])

        Q_lqr = np.array([[1.0]])
        R_lqr = np.array([[1.0]])
        Q_lqe = np.array([[1.0]])
        R_lqe = np.array([[1.0]])

        lqg = LQG(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe)

        self.assertIsNotNone(lqg.K)
        self.assertIsNotNone(lqg.L)

        x = np.array([1.0])
        lqg.x_hat = np.array([0.0])

        dt = 0.01
        u = np.array([0.0])
        y = C @ x
        for _ in range(500):
            u = lqg.compute_control(y=y, dt=dt, u_last=u)
            x_dot = A @ x + B @ u
            x = x + x_dot * dt
            y = C @ x

        self.assertLess(np.abs(x[0]), 1.0)
        self.assertLess(np.abs(x[0] - lqg.x_hat[0]), 0.1)


class TestH2Controller(unittest.TestCase):
    def test_h2_controller_solve(self):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        Q_lqr = np.array([[1.0]])
        R_lqr = np.array([[1.0]])
        Q_lqe = np.array([[1.0]])
        R_lqe = np.array([[1.0]])

        h2 = H2Controller(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe)
        K, L = h2.solve()

        self.assertIsNotNone(K)
        self.assertIsNotNone(L)

    def test_h2_controller_specific(self):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        Q_lqr = np.array([[1.0]])
        R_lqr = np.array([[1.0]])
        Q_lqe = np.array([[1.0]])
        R_lqe = np.array([[1.0]])

        controller = H2Controller(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe)
        K, L = controller.solve()

        self.assertTrue(K[0, 0] > 0)
        self.assertTrue(L[0, 0] > 0)


class TestFiniteHorizonLQR(unittest.TestCase):
    def test_finite_horizon_lqr(self):
        def A_fn(t):
            return np.array([[0, 1], [0, 0]])

        def B_fn(t):
            return np.array([[0], [1]])

        def Q_fn(t):
            return np.eye(2)

        def R_fn(t):
            return np.eye(1)

        Pf = np.eye(2) * 10.0
        T = 2.0

        fhlqr = FiniteHorizonLQR(A_fn, B_fn, Q_fn, R_fn, Pf, T)
        t_span, P_traj = fhlqr.solve(num_points=10)

        self.assertEqual(len(t_span), 10)
        self.assertEqual(P_traj.shape, (10, 2, 2))

        K0 = fhlqr.get_gain(0.0)
        self.assertEqual(K0.shape, (1, 2))

        KT = fhlqr.get_gain(2.0)
        np.testing.assert_allclose(KT, [[0.0, 10.0]], atol=1e-5)

    def test_finite_horizon_lqr_compute_control(self):
        def A_fn(t):
            return np.array([[0, 1], [0, 0]])

        def B_fn(t):
            return np.array([[0], [1]])

        def Q_fn(t):
            return np.eye(2)

        def R_fn(t):
            return np.eye(1)

        Pf = np.eye(2) * 10.0
        T = 2.0
        fhlqr = FiniteHorizonLQR(A_fn, B_fn, Q_fn, R_fn, Pf, T)
        u = fhlqr.compute_control(x=np.array([1.0, 0.0]), t=0.0)
        self.assertEqual(u.shape, (1,))


class TestHInfinity(unittest.TestCase):
    def test_h_infinity_scalar(self):
        A = np.array([[1.0]])
        B1 = np.array([[1.0]])
        B2 = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        gamma = 2.0

        hinf = HInfinityController(A, B1, B2, Q, R, gamma)
        K = hinf.compute_gain()

        self.assertLess(A[0, 0] - B2[0, 0] * K[0, 0], 0)

    def test_h_infinity_fail(self):
        A = np.array([[1.0]])
        B1 = np.array([[10.0]])
        B2 = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        gamma = 0.1

        hinf = HInfinityController(A, B1, B2, Q, R, gamma)
        with self.assertRaises(ValueError):
            hinf.solve()

    def test_h_infinity_compute_control(self):
        A = np.array([[1.0]])
        B1 = np.array([[1.0]])
        B2 = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        gamma = 2.0
        hinf = HInfinityController(A, B1, B2, Q, R, gamma)
        u = hinf.compute_control(np.array([1.0]))
        self.assertEqual(len(u), 1)


class TestNewControllers(unittest.TestCase):
    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc(self):
        nx = 2
        nu = 1
        horizon = 10
        dt = 0.1

        def dynamics(x, u):
            return ca.vertcat(x[1], u[0])

        def cost(x, u):
            return x[0] ** 2 + x[1] ** 2 + 0.1 * u[0] ** 2

        def terminal_cost(x):
            return 10.0 * (x[0] ** 2 + x[1] ** 2)

        nmpc = CasadiNMPC(
            nx=nx,
            nu=nu,
            horizon=horizon,
            dt=dt,
            dynamics_func=dynamics,
            cost_func=cost,
            terminal_cost_func=terminal_cost,
            u_min=[-10.0],
            u_max=[10.0],
            discrete=False,
        )

        x0 = np.array([1.0, 0.0])
        U = nmpc.solve(x0)

        self.assertEqual(U.shape, (horizon, nu))
        self.assertLess(U[0, 0], 0)

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_1d(self):
        nx = 1
        nu = 1
        horizon = 3
        dt = 0.1

        def dynamics(x, u):
            return u

        def stage_cost(x, u):
            return x**2 + u**2

        def term_cost(x):
            return x**2

        nmpc = CasadiNMPC(
            nx=nx,
            nu=nu,
            horizon=horizon,
            dt=dt,
            dynamics_func=dynamics,
            cost_func=stage_cost,
            terminal_cost_func=term_cost,
            u_min=-1.0,
            u_max=1.0,
            x_min=-10.0,
            x_max=10.0,
            discrete=False,
        )

        u_opt = nmpc.solve(x0=np.array([1.0]))
        self.assertEqual(u_opt.shape, (horizon, nu))
        self.assertLess(u_opt[0, 0], 0)

    def test_geometric_control(self):
        J = np.diag([1.0, 2.0, 3.0])
        kR = 10.0
        kW = 2.0
        ctrl = GeometricController(J, kR, kW)

        R = np.eye(3)
        omega = np.zeros(3)
        R_d = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        omega_d = np.zeros(3)

        M = ctrl.compute_control(R, omega, R_d, omega_d)
        self.assertEqual(len(M), 3)
        self.assertGreater(M[2], 0)

        M_acc = ctrl.compute_control(R, omega, R_d, omega_d, d_omega_d=np.zeros(3))
        self.assertEqual(len(M_acc), 3)

    def test_passivity_control(self):
        M = lambda q: np.eye(2)
        C = lambda q, q_dot: np.zeros((2, 2))
        G = lambda q: np.zeros(2)

        K_d = np.eye(2)
        Lambda = np.eye(2)
        ctrl = PassivityBasedController(M, C, G, K_d, Lambda)

        u = ctrl.compute_control([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        self.assertEqual(len(u), 2)
        self.assertLess(u[0], 0)

        u_acc = ctrl.compute_control(
            [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], q_ddot_d=np.zeros(2)
        )
        self.assertEqual(len(u_acc), 2)

    def test_backstepping_control(self):
        f = lambda x1, x2: np.zeros_like(x1)
        g = lambda x1, x2: np.eye(1)
        ctrl = BacksteppingController(f, g, k1=2.0, k2=1.0)

        u = ctrl.compute_control([1.0], [0.0], [0.0], [0.0])
        self.assertEqual(len(u), 1)
        self.assertAlmostEqual(u[0], -3.0)

        u_acc = ctrl.compute_control([1.0], [0.0], [0.0], [0.0], x1_ddot_d=[0.0])
        self.assertEqual(len(u_acc), 1)

    def test_backstepping_control_matrix_g(self):
        f = lambda x1, x2: np.zeros(2)
        g = lambda x1, x2: np.array([[1.0], [0.0]])
        k1 = np.eye(2)
        k2 = np.eye(2)
        ctrl = BacksteppingController(f, g, k1=k1, k2=k2)
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 0.0])
        x1_d = np.array([0.0, 0.0])
        x1_dot_d = np.array([0.0, 0.0])
        u = ctrl.compute_control(x1, x2, x1_d, x1_dot_d)
        self.assertEqual(len(u), 1)

    def test_mrac(self):
        mrac = ModelReferenceAdaptiveControl(
            [[-1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], lambda x: np.array([x[0]])
        )
        u = mrac.compute_control([1.0], [1.0], [1.0])
        self.assertEqual(len(u), 1)
        mrac.update_theta(0.1)

    def test_indi(self):
        ctrl = INDIController(lambda x, x_dot: np.eye(1))
        u = ctrl.compute_control([1.0], [2.0], [1.0], [0], [0])
        self.assertEqual(len(u), 1)
        self.assertAlmostEqual(u[0], 0.0)

    def test_indi_matrix_g(self):
        g = lambda x, x_dot: np.array([[1.0, 0.0], [0.0, 1.0]])
        ctrl = INDIController(g)
        u = ctrl.compute_control([1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [0, 0], [0, 0])
        self.assertEqual(len(u), 2)

    def test_indi_outer_loop_pd(self):
        pd = INDIOuterLoopPD(Kp=1.0, Kd=1.0)
        v = pd.compute_v([1], [0], [0], [0])
        self.assertEqual(v[0], -1.0)

        pd_mat = INDIOuterLoopPD(Kp=np.eye(2), Kd=np.eye(2))
        v_mat = pd_mat.compute_v([1, 0], [0, 0], [0, 0], [0, 0], x_ddot_d=[0, 0])
        self.assertEqual(v_mat[0], -1.0)

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_discrete(self):
        def f(x, u):
            return ca.vertcat(x[1], u[0])

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc_disc = CasadiNMPC(2, 1, 5, 0.1, f, L, V, discrete=True)
        u = nmpc_disc.solve([1, 0])
        self.assertEqual(u.shape, (5, 1))

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_bounds_scalar(self):
        def f(x, u):
            return ca.vertcat(x[1], u[0])

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = CasadiNMPC(2, 1, 5, 0.1, f, L, V, u_min=1.0)
        u = nmpc.solve([1, 0], u_guess=[1, 1, 1, 1, 1])
        self.assertEqual(u.shape, (5, 1))

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_bounds_mismatch(self):
        def f(x, u):
            return ca.vertcat(x[1], u[0])

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        with self.assertRaises(ValueError):
            nmpc = CasadiNMPC(2, 1, 5, 0.1, f, L, V, u_min=[1, 2, 3])

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_guess_mismatch(self):
        def f(x, u):
            return ca.vertcat(x[1], u[0])

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = CasadiNMPC(2, 1, 5, 0.1, f, L, V)
        u = nmpc.solve([1, 0], u_guess=[1, 1])
        self.assertEqual(u.shape, (5, 1))

    @unittest.skipIf(ca is None, "CasADi not installed")
    def test_casadi_nmpc_bounds_array(self):
        def f(x, u):
            return ca.vertcat(x[1], u[0] + u[1])

        def L(x, u):
            return x[0] ** 2

        def V(x):
            return x[0] ** 2

        nmpc = CasadiNMPC(2, 2, 5, 0.1, f, L, V, u_min=[1.0, 2.0])
        self.assertEqual(len(nmpc.u_min), 2)


if __name__ == "__main__":
    unittest.main()
