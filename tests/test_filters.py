import numpy as np
import pytest
from opengnc.kalman_filters.kf import KF
from opengnc.kalman_filters.ekf import EKF
from opengnc.kalman_filters.mekf import MEKF
from opengnc.kalman_filters.ukf import UKF, UKF_Attitude
from opengnc.kalman_filters import SRUKF, EnKF, CKF, ParticleFilter, AKF, IMM
from opengnc.kalman_filters.fixed_interval_smoother import fixed_interval_smoother
from unittest.mock import patch

from opengnc.utils.quat_utils import quat_mult, quat_normalize, axis_angle_to_quat, quat_rot, quat_conj
from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.sun_sensor import SunSensor

def test_kf_initialization():
    dim_x = 2
    dim_z = 1
    kf = KF(dim_x, dim_z)
    assert kf.x.shape == (dim_x,)
    assert kf.P.shape == (dim_x, dim_x)
    assert kf.F.shape == (dim_x, dim_x)
    assert kf.H.shape == (dim_z, dim_x)
    assert kf.R.shape == (dim_z, dim_z)
    assert kf.Q.shape == (dim_x, dim_x)

def test_kf_constant_velocity():
    """Test KF with a simple 1D constant velocity model."""
    dt = 0.1
    kf = KF(dim_x=2, dim_z=1)
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 10
    kf.R *= 0.1
    kf.Q = np.array([[0.001, 0], [0, 0.001]])

    true_x = np.array([0., 1.0]) # Pos=0, Vel=1 m/s

    for _ in range(20):
        true_x = np.dot(kf.F, true_x)
        z = np.dot(kf.H, true_x) + np.random.normal(0, np.sqrt(kf.R[0,0]))
        kf.predict()
        kf.update(z)
        
    error_pos = np.abs(true_x[0] - kf.x[0])
    error_vel = np.abs(true_x[1] - kf.x[1])
    
    assert error_pos < 1.0
    assert error_vel < 0.5

def test_ekf_nonlinear_tracking():
    dt = 0.1
    ekf = EKF(dim_x=1, dim_z=1)
    
    def f_func(x, dt, u, **kwargs): return x + 1.0 * dt
    def f_jac(x, dt, u, **kwargs): return np.array([[1.0]])
    
    def h_func(x, **kwargs): return x**2
    def h_jac(x, **kwargs): return np.array([[2*x[0]]])
    
    ekf.P *= 1.0
    ekf.R *= 0.01
    ekf.Q *= 0.001
    ekf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ekf.predict(f_func, f_jac, dt)
        ekf.update(np.array([z]), h_func, h_jac)
        
    error = np.abs(true_x - ekf.x[0])
    assert error < 0.5

def test_mekf_attitude_tracking():
    # Set seed for determinism in random sensor noise
    np.random.seed(42)
    dt = 0.1
    omega_body = np.array([0.1, 0.05, 0.2]) # Constant body rate
    z_ref_inertial = np.array([1.0, 0.0, 0.0])
    
    q0 = np.array([0, 0, 0, 1.0])
    beta0 = np.array([0.01, -0.01, 0.02])
    mekf = MEKF(q_init=q0, beta_init=beta0)
    
    mekf.Q = np.eye(6) * 0.0001
    mekf.R = np.eye(3) * 0.001 # Robust tuning for sensor std=0.01
    
    q_true = q0.copy()
    
    sun_sensor = SunSensor(noise_std=0.01)
    gyro = Gyroscope(noise_std=0.0, dt=dt)

    # Run for 100 steps to allow bias convergence
    for _ in range(100):
        dq_true = axis_angle_to_quat(omega_body * dt)
        q_true = quat_normalize(quat_mult(q_true, dq_true))
        
        z_body_true = quat_rot(quat_conj(q_true), z_ref_inertial)
        z_body_meas = sun_sensor.measure(z_body_true)
        z_body_meas.metadata["reference"] = z_ref_inertial

        gyro_packet = gyro.measure(omega_body)
        mekf.predict(gyro_packet)
        mekf.update(z_body_meas)
        
    error = 1.0 - np.abs(np.dot(q_true, mekf.q))
    assert error < 1e-3

def test_ukf_nonlinear_tracking():
    dt = 0.1
    ukf = UKF(dim_x=1, dim_z=1)
    
    def f_func(x, dt): return x + 1.0 * dt
    
    def h_func(x): return x**2
    
    ukf.P *= 1.0
    ukf.R *= 0.01
    ukf.Q *= 0.001
    ukf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ukf.predict(dt, f_func)
        ukf.update(np.array([z]), h_func)
        
    error = np.abs(true_x - ukf.x[0])
    assert error < 0.5

def test_ukf_attitude_initialization():
    ukf = UKF_Attitude()
    assert ukf.x.shape == (7,) 
    assert ukf.P.shape == (6, 6) 
    
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)

def test_ukf_attitude_prediction():
    ukf = UKF_Attitude()
    dt = 0.1

    ukf.x = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1])
    gyro = Gyroscope(noise_std=0.0, dt=dt)
    omega_packet = gyro.measure(np.array([0.1, 0.1, 0.1]))

    ukf.predict(omega_packet)

    assert np.allclose(ukf.x[:4], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5)
    assert np.allclose(ukf.x[4:], np.array([0.1, 0.1, 0.1]), atol=1e-5)

def test_ukf_attitude_update():
    ukf = UKF_Attitude()
    ukf.P *= 0.1
    ukf.R *= 0.001
    
    angle = np.pi/2
    q_true = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    z_ref = np.array([1.0, 0.0, 0.0])
    
    sensor = SunSensor(noise_std=0.0)
    
    q_conj = quat_conj(q_true)
    z_body_true = quat_rot(q_conj, z_ref)
    
    z_meas = sensor.measure(z_body_true)
    z_meas.metadata["reference"] = z_ref
    ukf.update(z_meas)
    
    assert ukf.x.shape == (7,)
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)

def test_rts_smoother():
    from opengnc.kalman_filters.rts_smoother import rts_smoother
    
    dt = 0.5
    num_steps = 50
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.5
    
    true_x = np.zeros((num_steps, 2))
    z = np.zeros(num_steps)
    
    x = np.array([0.0, 1.0])
    for k in range(num_steps):
        true_x[k] = x
        z[k] = np.dot(H, x).item() + np.random.normal(0, np.sqrt(R[0,0]))
        x = np.dot(F, x) + np.random.multivariate_normal(np.zeros(2), Q)
        
    kf = KF(dim_x=2, dim_z=1)
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R
    
    xs_filt = []
    ps_filt = []
    
    for k in range(num_steps):
        kf.predict()
        kf.update(z[k])
        xs_filt.append(kf.x.copy())
        ps_filt.append(kf.P.copy())
        
    Fs = [F] * (num_steps - 1)
    Qs = [Q] * (num_steps - 1)
    xs_smooth, ps_smooth = rts_smoother(xs_filt, ps_filt, Fs, Qs)
    
    rmse_filt = np.sqrt(np.mean((np.array(xs_filt)[:, 0] - true_x[:, 0])**2))
    rmse_smooth = np.sqrt(np.mean((xs_smooth[:, 0] - true_x[:, 0])**2))    
    assert rmse_smooth < rmse_filt
    assert xs_smooth.shape == (num_steps, 2)

def test_kf_with_control():
    kf = KF(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1], [0, 1]])
    kf.B = np.array([[0.5], [1.0]])
    u = np.array([1.0])
    kf.predict(u=u)
    assert np.allclose(kf.x, [0.5, 1.0])

def test_mekf_default_init():
    mekf = MEKF()
    assert np.allclose(mekf.q, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mekf.beta, [0.0, 0.0, 0.0])

def linear_fx(x, dt):
    F = np.array([[1, dt], [0, 1]])
    return np.dot(F, x)

def linear_hx(x):
    H = np.array([[1, 0]])
    return np.dot(H, x)

@pytest.fixture
def linear_setup():
    dt = 0.1
    dim_x = 2
    dim_z = 1
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.1
    Q = np.eye(2) * 0.01
    R = np.array([[0.05]])
    return dt, dim_x, dim_z, x0, P0, Q, R

def test_sr_ukf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = SRUKF(dim_x, dim_z)
    filter.x = x0.copy()
    filter.S = np.linalg.cholesky(P0)
    filter.Qs = np.linalg.cholesky(Q)
    filter.Rs = np.linalg.cholesky(R)
    
    filter.predict(dt, linear_fx)
    assert filter.x[0] == pytest.approx(0.1, abs=1e-5)
    
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert np.all(np.isfinite(filter.P))

def test_enkf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = EnKF(dim_x, dim_z, ensemble_size=20)
    filter.initialize_ensemble(x0, P0)
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert filter.X.shape == (2, 20)
    assert filter.P.shape == (2, 2)

def test_ckf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = CKF(dim_x, dim_z)
    filter.x = x0.copy()
    filter.P = P0.copy()
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2

def test_ckf_non_psd_fallback():
    filter = CKF(dim_x=2, dim_z=1)
    P_non_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
    x = np.array([0.0, 0.0])
    
    points = filter._generate_cubature_points(x, P_non_psd)
    assert points is not None
    assert points.shape == (2, 4)

def test_pf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = ParticleFilter(dim_x, dim_z, num_particles=100)
    filter.initialize_particles(x0, P0)
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert filter.particles.shape == (100, 2)
    assert filter.P.shape == (2, 2)
    
    filter.resample()
    assert np.allclose(filter.weights, 1.0 / 100)

def test_pf_automatic_resampling(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = ParticleFilter(dim_x, dim_z, num_particles=100)
    filter.initialize_particles(x0, P0)
    
    with patch.object(filter, 'neff', return_value=1.0):
        filter.update(np.array([0.15]), linear_hx)
    assert np.allclose(filter.weights, 1.0 / 100)

def test_akf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = AKF(dim_x, dim_z, window_size=5)
    filter.x = x0.copy()
    filter.P = P0.copy()
    filter.F = np.array([[1, dt], [0, 1]])
    filter.H = np.array([[1, 0]])
    
    for _ in range(6):
        filter.predict()
        filter.update(np.array([0.1]))
        
    assert len(filter.x) == 2
    assert filter.R.shape == (1, 1)

def test_imm(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    
    f1 = KF(dim_x, dim_z)
    f1.x = x0.copy()
    f1.P = P0.copy()
    f1.F = np.eye(2)
    f1.H = np.array([[1, 0]])
    
    f2 = KF(dim_x, dim_z)
    f2.x = x0.copy()
    f2.P = P0.copy()
    f2.F = np.array([[1, dt], [0, 1]])
    f2.H = np.array([[1, 0]])
    
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    imm = IMM([f1, f2], trans)
    
    imm.predict(dt)
    imm.update(np.array([0.1]))
    
    assert len(imm.x) == 2
    assert len(imm.mu) == 2
    assert np.isclose(np.sum(imm.mu), 1.0)

def test_imm_with_nonlinear_filter():
    f1 = CKF(dim_x=2, dim_z=1)
    f2 = CKF(dim_x=2, dim_z=1)
    
    trans = np.array([[0.9, 0.1], [0.1, 0.9]])
    imm = IMM([f1, f2], trans)
    
    dt = 0.1
    def linear_fx(x, dt):
        return x + np.array([1.0, 0.0]) * dt
        
    def linear_hx(x):
        return np.array([x[0]])
        
    imm.predict(dt, fx=linear_fx)
    z = np.array([0.5])
    imm.update(z, hx=linear_hx)
    
    assert len(imm.x) == 2
    assert len(imm.mu) == 2

def test_sr_ukf_cholesky_update_edge_cases():
    filter = SRUKF(dim_x=2, dim_z=1)
    S = np.eye(2)
    v = np.array([1.0, 0.0])
    
    S_up = filter._cholesky_update(S, v, 1.0)
    assert S_up is not None
    assert S_up.shape == (2, 2)
    
    S_down = filter._cholesky_update(S, v, -1.0)
    assert S_down is not None
    assert S_down.shape == (2, 2)

def test_ukf_non_psd_fallback():
    filter = UKF(dim_x=2, dim_z=1)
    P_non_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
    x = np.array([0.0, 0.0])
    
    points = filter.generate_sigma_points(x, P_non_psd)
    assert points is not None
    assert points.shape == (5, 2) 

def test_fixed_interval_smoother():
    dt = 0.5
    num_steps = 30
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    
    true_x = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 1))
    
    x = np.array([0.0, 1.0])
    for k in range(num_steps):
        true_x[k] = x
        z[k] = np.dot(H, x) + np.random.normal(0, np.sqrt(R[0,0]))
        x = np.dot(F, x) + np.random.multivariate_normal(np.zeros(2), Q)
        
    kf = KF(dim_x=2, dim_z=1)
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R
    
    xs_filt = []
    ps_filt = []
    
    for k in range(num_steps):
        kf.predict()
        kf.update(z[k])
        xs_filt.append(kf.x.copy())
        ps_filt.append(kf.P.copy())
        
    Fs = [F] * (num_steps - 1)
    Qs = [Q] * (num_steps - 1)
    Zs = [zi for zi in z]
    Hs = [H] * num_steps
    Rs = [R] * num_steps
    
    xs_smooth, ps_smooth = fixed_interval_smoother(xs_filt, ps_filt, Fs, Qs, Zs, Hs, Rs)
    
    rmse_filt = np.sqrt(np.mean((np.array(xs_filt)[:, 0] - true_x[:, 0])**2))
    rmse_smooth = np.sqrt(np.mean((xs_smooth[:, 0] - true_x[:, 0])**2))
    
    assert rmse_smooth <= rmse_filt * 1.05
    assert xs_smooth.shape == (num_steps, 2)
    
    for k in range(num_steps):
        assert np.trace(ps_smooth[k]) <= np.trace(ps_filt[k]) * 1.01

def test_ukf_attitude_branches():
    ukf = UKF_Attitude()
    q1 = np.array([0, 0, 0, 1.0])
    q2 = np.array([0, 0, 0, -1.0])
    x1 = np.concatenate([q1, np.zeros(3)])
    x2 = np.concatenate([q2, np.zeros(3)])
    dx = ukf.subtract_x(x1, x2)
    assert np.allclose(dx, 0.0)
    sigmas = np.array([
        [0, 0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, -1.0, 0, 0, 0]
    ])
    weights = np.array([0.5, 0.5])
    x_mean = ukf.mean_x(sigmas, weights)
    assert np.allclose(x_mean[:4], np.array([0, 0, 0, 1.0]))

def test_imm_mu_setter():
    from opengnc.kalman_filters.ekf import EKF
    kf1 = EKF(3, 3)
    kf2 = EKF(3, 3)
    imm = IMM([kf1, kf2], np.eye(2))
    imm.mu = np.array([0.7, 0.3])
    assert np.allclose(imm.mu_probs, [0.7, 0.3])

def test_imm_predict_dt_branch():
    class MockFilter:
        def __init__(self):
            self.dim_x = 3
            self.x = np.zeros(3)
            self.P = np.eye(3)
        def predict(self, dt, **kwargs):
            self.predict_called_with_dt = dt
    mf = MockFilter()
    imm = IMM([mf], np.array([[1.0]]))
    imm.predict(dt=10.0)
    assert mf.predict_called_with_dt == 10.0




