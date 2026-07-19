import pytest
import numpy as np
from opengnc.sensors.star_tracker import StarTracker
from opengnc.sensors.sun_sensor import SunSensor
from opengnc.sensors.magnetometer import Magnetometer
from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.gnss_receiver import GNSSReceiver
from opengnc.sensors.imu import Accelerometer, IMU
from opengnc.sensors.sun_sensor_array import CoarseSunSensorArray
from opengnc.sensors.horizon_sensor import HorizonSensor
from opengnc.sensors.altimeter import Altimeter
from opengnc.sensors.lidar import Lidar
from opengnc.sensors.camera import Camera
from opengnc.sensors.star_catalog import StarCatalog
from opengnc.sensors.sensor import Sensor

from opengnc.actuators.reaction_wheel import ReactionWheel
from opengnc.actuators.magnetorquer import Magnetorquer
from opengnc.actuators.thruster import Thruster, ChemicalThruster, ElectricThruster
from opengnc.actuators.cmg import ControlMomentGyro
from opengnc.actuators.vscmg import VariableSpeedCMG
from opengnc.actuators.solar_sail import SolarSail
from opengnc.actuators.allocation import PseudoInverseAllocator, SingularRobustAllocator, NullMotionManager


class TestSensors:
    def test_star_tracker(self):
        st = StarTracker(noise_std=0.0, bias=np.array([0.1, 0, 0]))
        true_quat = np.array([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]
        measured = st.measure(true_quat).value
        
        assert not np.allclose(measured, true_quat)
        assert measured[0] > 0
        
        st_noisy = StarTracker(noise_std=0.01)
        m1 = st_noisy.measure(true_quat).value
        m2 = st_noisy.measure(true_quat).value
        assert not np.allclose(m1, m2)

    def test_sun_sensor(self):
        ss = SunSensor(noise_std=0.0)
        vec = np.array([1.0, 0.0, 0.0])
        meas = ss.measure(vec).value
        assert np.allclose(meas, vec)
        
        ss_noisy = SunSensor(noise_std=0.1)
        meas_noisy = ss_noisy.measure(vec).value
        assert np.isclose(np.linalg.norm(meas_noisy), 1.0)

    def test_magnetometer(self):
        mag = Magnetometer(bias=np.array([1e-6, 0, 0]))
        true_b = np.array([20e-6, 0, 0])
        meas = mag.measure(true_b).value
        assert np.allclose(meas, np.array([21e-6, 0, 0]))

    def test_gyroscope(self):
        gyro = Gyroscope(noise_std=0.0, bias_stability=0.01, dt=1.0)
        w_true = np.zeros(3)
        
        b0 = gyro.current_bias.copy()
        m1 = gyro.measure(w_true).value
        b1 = gyro.current_bias.copy()
        
        assert not np.allclose(b0, b1)
        assert np.allclose(m1, b1)

    def test_gnss_receiver(self):
        gnss = GNSSReceiver(pos_noise_std=0.0, vel_noise_std=0.0, 
                             pos_bias=np.array([1, 2, 3]))
        r_true = np.array([7000e3, 0, 0])
        v_true = np.array([0, 7.5e3, 0])
        gnss_packet = gnss.measure(r_true, v_true)
        r_meas, v_meas = gnss_packet.value[:3], gnss_packet.value[3:]
        assert np.allclose(r_meas, r_true + np.array([1, 2, 3]))
        assert np.allclose(v_meas, v_true)

    def test_accelerometer(self):
        acc = Accelerometer(noise_std=0.0, bias=np.array([0.1, 0, 0]), scale_factor=1.1)
        a_true = np.array([1.0, 0.0, 0.0])
        a_meas = acc.measure(a_true).value
        assert np.isclose(a_meas[0], 1.2)

    def test_imu(self):
        imu = IMU(gyro_params={'noise_std': 0.0}, accel_params={'noise_std': 0.0})
        w_true = np.array([0.1, 0, 0])
        a_true = np.array([0, 0, 9.8])
        imu_packet = imu.measure(w_true, a_true)
        w_meas, a_meas = imu_packet.value[:3], imu_packet.value[3:]
        assert np.allclose(w_meas, w_true)
        assert np.allclose(a_meas, a_true)

    def test_css_array(self):
        css = CoarseSunSensorArray(noise_std=0.0)
        sun_vec = np.array([1.0, 0.0, 0.0])
        meas = css.measure(sun_vec).value
        assert np.isclose(meas[0], 1.0)
        assert np.all(meas[1:] == 0.0)

    def test_horizon_sensor(self):
        hs = HorizonSensor(noise_std=0.0)
        nadir = np.array([0, 0, 1.0])
        meas = hs.measure(nadir).value
        assert np.allclose(meas, nadir)

    def test_altimeter(self):
        alt = Altimeter(noise_std=0.0, bias=10.0)
        meas = alt.measure(500e3).value.item()
        assert np.isclose(meas, 500010.0)

    def test_lidar(self):
        lidar = Lidar(range_noise_std=0.0, los_noise_std=0.0)
        rel_pos = np.array([10.0, 0.0, 0.0])
        lidar_packet = lidar.measure(rel_pos)
        r_meas, los_meas = lidar_packet.value[0], lidar_packet.value[1:]
        assert np.isclose(r_meas, 10.0)
        assert np.allclose(los_meas, np.array([1.0, 0.0, 0.0]))

    def test_camera(self):
        cam = Camera(focal_length=1.0, resolution=(1000, 1000), sensor_size=(1.0, 1.0))
        p = np.array([0, 0, 10.0])
        uv = cam.measure(p).value
        assert np.allclose(uv, np.array([500, 500]))
        p2 = np.array([0.1, 0, 10.0])
        uv2 = cam.measure(p2).value
        assert np.allclose(uv2, np.array([510, 500]))

    def test_star_catalog(self):
        cat = StarCatalog()
        cat.load_catalog(None)
        stars = cat.get_stars_in_fov(np.array([1, 0, 0]), 10.0)
        assert len(stars) == 1
        assert stars[0]['id'] == 1

    def test_star_catalog_init_with_path(self):
        cat = StarCatalog("dummy_path")
        assert len(cat.stars) == 4

    def test_camera_fov_noise(self):
        cam = Camera(noise_std=1.0)
        p = np.array([0, 0, 10.0])
        uv = cam.measure(p).value
        p_back = np.array([0, 0, -10.0])
        with pytest.raises(ValueError, match="behind the camera"):
            cam.measure(p_back)
        p_outside = np.array([10.0, 10.0, 1.0])
        with pytest.raises(ValueError, match="outside the image plane"):
            cam.measure(p_outside)

    def test_sensor_base_features(self):
        class DummySensor(Sensor):
            def measure(self, true_state):
                return self.build_measurement(true_state)
        s = DummySensor()
        res_cal = s.apply_calibration(np.array([1,2,3]), scale_factor=np.array([2,2,2]))
        assert np.allclose(res_cal, [2,4,6])
        res_cal2 = s.apply_calibration(5.0, scale_factor=2.0, bias=1.0)
        assert res_cal2 == 11.0
        res_fogm = s.apply_fogm_noise(1.0, 0.0, 10.0, 1.0)
        assert res_fogm == 1.0
        s.fault_state = "noise_increase"
        s.apply_faults(np.array([1.0, 2.0]))

    def test_star_catalog_min_mag(self):
        cat = StarCatalog()
        cat.load_catalog(None)
        stars = cat.get_stars_in_fov(np.array([1,0,0]), 10.0, min_mag=-1.0)
        assert len(stars) == 0

    def test_star_tracker_small_angle(self):
        st = StarTracker(noise_std=0.0)
        q = np.array([0, 0, 0, 1.0])
        q_m = st.measure(q).value
        assert np.allclose(q_m, q)

    def test_horizon_sensor_bias(self):
        hs = HorizonSensor(noise_std=0.0, bias=np.array([0.1, 0.1]))
        nadir = np.array([0, 0, 1.0])
        meas = hs.measure(nadir).value
        assert meas[0] != 0.0

    def test_lidar_zero_range(self):
        lidar = Lidar()
        lidar_packet = lidar.measure(np.zeros(3))
        r_meas, los_meas = lidar_packet.value[0], lidar_packet.value[1:]
        assert r_meas >= 0
        assert np.allclose(los_meas, np.zeros(3))

    def test_sun_sensor_array_coverage(self):
        ssa = CoarseSunSensorArray(boresights=[np.array([1, 0, 0])])
        assert len(ssa.boresights) == 1
        res = ssa.measure(np.array([1, 0, 0])).value
        assert res.shape == (1,)

class TestActuators:
    def test_rw_friction(self):
        # RW with friction
        rw = ReactionWheel(max_torque=1.0, static_friction=0.1, viscous_friction=0.01)
        
        # Test static friction at zero speed
        # Command < static friction should result in 0
        assert rw.command(0.05, current_speed=0.0) == 0.0
        # Command > static friction
        assert rw.command(0.5, current_speed=0.0) == 0.4
        
        # Test viscous friction at speed
        # Command 0.5 at speed 10 -> Friction = 0.01 * 10 = 0.1. Result = 0.5 - 0.1 = 0.4
        assert rw.command(0.5, current_speed=10.0) == 0.4
    
    def test_cmg_torque(self):
        # CMG: h along Z, gimbal along X -> Torque should be along Y
        cmg = ControlMomentGyro(wheel_momentum=10.0, gimbal_axis=[1, 0, 0], spin_axis_init=[0, 0, 1])
        
        # gimbal rate = 0.1 rad/s
        # T = g_rate * h * (g x s) = 0.1 * 10 * ([1,0,0] x [0,0,1]) = 1.0 * [0, -1, 0]
        torque = cmg.command(0.1)
        np.testing.assert_allclose(torque, [0, -1.0, 0], atol=1e-7)
    
    def test_solar_sail_force(self):
        sail = SolarSail(area=100.0, reflectivity=1.0, specular_reflect_coeff=1.0) # Perfect specular reflection
        
        # Sun along X, Normal along X
        sun_vec = np.array([1, 0, 0])
        normal = np.array([1, 0, 0])
        
        # F = P * A * cos(theta) * ( (1-rho)*u + 2*rho*cos(theta)*n )
        # cos_theta = 1, rho = 1 -> F = P * A * (0*u + 2*1*n) = 2 * P * A * n
        # P = 4.56e-6
        force = sail.calculate_force(sun_vec, normal)
        expected_f = 2 * 4.56e-6 * 100.0 * np.array([1, 0, 0])
        np.testing.assert_allclose(force, expected_f, atol=1e-10)
    
    def test_pseudo_inverse_allocation(self):
        # 4 RWs in pyramid configuration
        # Angle beta = 54.7 deg
        beta = np.deg2rad(54.744)
        c, s = np.cos(beta), np.sin(beta)
        
        # Rows: Tx, Ty, Tz
        A = np.array([
            [s, 0, -s, 0],
            [0, s, 0, -s],
            [c, c, c, c]
        ])
        
        allocator = PseudoInverseAllocator(A)
        
        desired_torque = np.array([0.1, 0.0, 0.0])
        u = allocator.allocate(desired_torque)
        
        # Verify A * u == desired
        np.testing.assert_allclose(A @ u, desired_torque, atol=1e-7)
    
    def test_null_motion(self):
        A = np.array([[1, 1]]) # Two actuators for 1-DOF
        manager = NullMotionManager(A)
        
        u_base = np.array([0.5, 0.5]) # Producer 1.0 total
        z = np.array([1.0, -1.0]) # Desired shift
        
        u_net = manager.apply_null_command(u_base, z)
        
        # Total output should still be 1.0
        assert np.isclose(np.sum(u_net), 1.0)
        # The actuators should have shifted
        assert u_net[0] > 0.5
        assert u_net[1] < 0.5
    
    def test_actuator_base_features(self):
        from opengnc.actuators.actuator import Actuator
        
        class TestActuator(Actuator):
            def command(self, signal, **kwargs):
                return signal
                
        # Test Saturation with Tuple (min, max)
        act = TestActuator(name="Test", saturation=(-1.0, 5.0))
        assert act.apply_saturation(10.0) == 5.0
        assert act.apply_saturation(-5.0) == -1.0
        assert act.apply_saturation(2.0) == 2.0
        
        # Test Saturation with Single Float/Int (magnitude)
        act2 = TestActuator(name="Test2", saturation=3.0)
        assert act2.apply_saturation(5.0) == 3.0
        assert act2.apply_saturation(-5.0) == -3.0
        
        # Test Deadband
        act_db = TestActuator(deadband=1.0)
        assert act_db.apply_deadband(0.5) == 0.0
        assert act_db.apply_deadband(-0.5) == 0.0
        assert act_db.apply_deadband(1.5) == 1.5
        
        # Test Saturation with Invalid list length
        act3 = TestActuator(name="Test3", saturation=[1.0])
        assert act3.apply_saturation(2.0) == 2.0
    
    
    def test_singular_robust_allocator_grid(self):
        A2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # 2x3
        allocator2 = SingularRobustAllocator(A2, epsilon=0.5, lambda0=0.01)
        
        # Case 1: normal allocation
        u = allocator2.allocate(np.array([1.0, 0.0]))
        np.testing.assert_allclose(A2 @ u, [1.0, 0.0], atol=1e-7)
        
        # Case 2: trigger singularity
        A_near_sing = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
        allocator_sing = SingularRobustAllocator(A_near_sing, epsilon=0.5, lambda0=0.01)
        u_sing = allocator_sing.allocate(np.array([1.0, 1.0]))
        assert u_sing is not None
        
        # Case 3: A_current
        A_curr = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        u_curr = allocator2.allocate(np.array([1.0, 0.0]), A_current=A_curr)
        np.testing.assert_allclose(A_curr @ u_curr, [1.0, 0.0], atol=1e-7)
    
    def test_cmg_initialization_ortho(self):
        g = np.array([1, 0, 0])
        s = np.array([1, 1, 0]) / np.sqrt(2)
        cmg = ControlMomentGyro(wheel_momentum=10.0, gimbal_axis=g, spin_axis_init=s)
        
        assert abs(np.dot(cmg.g_axis, cmg.s_axis)) < 1e-6
        np.testing.assert_allclose(cmg.s_axis, [0, 1, 0], atol=1e-6)
        
        # Test command with dt and wrapping
        cmg.command(np.pi, dt=1.0)
        assert np.isclose(cmg.gimbal_angle, np.pi) or np.isclose(cmg.gimbal_angle, -np.pi)
        
        cmg.command(np.pi, dt=1.0)
        assert np.isclose(cmg.gimbal_angle, 0.0, atol=1e-6)
        
        jac = cmg.get_torque_jacobian()
        assert jac.shape == (3, 1)
    
    def test_rw_negative_momentum_saturation(self):
        rw = ReactionWheel(max_torque=1.0, max_momentum=10.0, inertia=1.0)
        assert rw.command(-0.5, current_speed=-10.0) == 0.0
        assert rw.command(0.5, current_speed=-10.0) == 0.5
    
    def test_solar_sail_backside(self):
        sail = SolarSail(area=100.0)
        sun_vec = np.array([-1, 0, 0])
        normal = np.array([1, 0, 0])
        
        force = sail.calculate_force(sun_vec, normal)
        np.testing.assert_allclose(force, [0, 0, 0])
        
        force_cmd = sail.command(normal, sun_vec=sun_vec, distance_au=1.0)
        np.testing.assert_allclose(force_cmd, [0, 0, 0])
    
    def test_thruster_mass_flow(self):
        thr = Thruster(isp=300.0)
        g0 = 9.80665
        expected_m_dot = 10.0 / (300.0 * g0)
        assert np.isclose(thr.get_mass_flow(10.0), expected_m_dot)
        
        thr_no_isp = Thruster(isp=None)
        assert thr_no_isp.get_mass_flow(10.0) == 0.0
    
    def test_chemical_thruster_pwm_limit(self):
        chem = ChemicalThruster(max_thrust=10.0, min_on_time=0.1)
        # Manually clear min_impulse_bit to bypass super().command check
        chem.min_impulse_bit = 0.0
        assert chem.command(1.0, dt=0.5) == 0.0
    
    
    def test_thruster_cluster_all(self):
        from opengnc.actuators.thruster import ThrusterCluster, ChemicalThruster
        
        thrusters = [ChemicalThruster(max_thrust=1.0) for _ in range(4)]
        positions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0]
        ]
        directions = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ]
        
        cluster = ThrusterCluster(thrusters, positions, directions)
        assert cluster.A.shape == (6, 4)
        
        cmd = np.array([0, 0, 1.0, 0, 0, 0])
        delivered = cluster.command(cmd, dt=0.1)
        assert np.isclose(np.sum(delivered), 1.0)
    
    def test_vscmg_all(self):
        vscmg = VariableSpeedCMG(wheel_inertia=1.0, gimbal_axis=[1,0,0], spin_axis_init=[0,0,1], max_wheel_torque=1.0)
        
        torque = vscmg.command((0.1, 0.5))
        np.testing.assert_allclose(torque, [0, 0, 0.5], atol=1e-7)
        
        vscmg.command((0.0, 1.0), dt=1.0)
        assert np.isclose(vscmg.wheel_speed, 1.0)
        
        torque2 = vscmg.command((0.1, 0.0))
        np.testing.assert_allclose(torque2, [0, -0.1, 0], atol=1e-7)
        
        jac = vscmg.get_jacobian()
        assert jac.shape == (3, 2)
        
        # Test without max_wheel_torque
        vscmg2 = VariableSpeedCMG(wheel_inertia=1.0, gimbal_axis=[1,0,0], spin_axis_init=[0,0,1], max_wheel_torque=None)
        torque2_no_limit = vscmg2.command((0.1, 5.0))
        np.testing.assert_allclose(torque2_no_limit, [0, 0, 5.0], atol=1e-7)
    
    def test_reaction_wheel(self):
        rw = ReactionWheel(max_torque=0.1, max_momentum=1.0, inertia=0.1)
        
        assert rw.command(0.05) == 0.05
        assert rw.command(0.2) == 0.1
        assert rw.command(-0.2) == -0.1
        assert rw.command(0.05, current_speed=10.0) == 0.0
        assert rw.command(-0.05, current_speed=10.0) == -0.05

    def test_magnetorquer(self):
        mtq = Magnetorquer(max_dipole=10.0)
        assert mtq.command(5.0) == 5.0
        assert mtq.command(15.0) == 10.0
        assert mtq.command(-12.0) == -10.0

    def test_thruster(self):
        thr = Thruster(max_thrust=5.0)
        assert thr.command(2.0) == 2.0
        assert thr.command(6.0) == 5.0
        
        thr_mib = Thruster(max_thrust=1.0, min_impulse_bit=0.5) 
        assert thr_mib.command(0.4, dt=1.0) == 0.0
        assert thr_mib.command(0.6, dt=1.0) == 0.6
        
        chem = ChemicalThruster(max_thrust=100.0, min_on_time=0.01)
        assert chem.command(150.0) == 100.0
        assert chem.command(5.0, dt=0.1) == 0.0
        assert chem.command(15.0, dt=0.1) == 15.0

    def test_electric_thruster(self):
        ethr = ElectricThruster(max_thrust=0.1, isp=1500, power_efficiency=0.5)
        p = ethr.get_power_consumption(0.1)
        assert 1400 < p < 1500

    def test_thruster_efficiency_zero(self):
        t = ElectricThruster(power_efficiency=0.0, isp=1500, max_thrust=0.1)
        assert t.get_power_consumption(0.1) == float('inf')
        
        t_neg = ElectricThruster(power_efficiency=-0.1, isp=1500, max_thrust=0.1)
        assert t_neg.get_power_consumption(0.1) == float('inf')





