import json
from pathlib import Path

import numpy as np

from opengnc.kalman_filters import AttitudeSensorFusion
from opengnc.sensors import (
    GNSSReceiver,
    Gyroscope,
    HorizonSensor,
    IMU,
    Lidar,
    Magnetometer,
    StarTracker,
    SunSensor,
    load_sensor_suite,
)
from opengnc.utils.quat_utils import axis_angle_to_quat, quat_conj, quat_mult, quat_normalize, quat_rot


def test_sensor_packet_interface_consistency():
    sensors_and_inputs = [
        (Gyroscope(noise_std=0.0), (np.array([0.1, -0.2, 0.3]),), "angular_rate", "rad/s", "body", 3),
        (SunSensor(noise_std=0.0), (np.array([1.0, 0.0, 0.0]),), "sun_vector", "unit_vector", "body", 3),
        (Magnetometer(noise_std=0.0), (np.array([2e-5, 0.0, -1e-5]),), "magnetic_field", "T", "body", 3),
        (HorizonSensor(noise_std=0.0), (np.array([0.0, 0.0, 1.0]),), "nadir_vector", "unit_vector", "body", 3),
        (StarTracker(noise_std=0.0), (np.array([0.0, 0.0, 0.0, 1.0]),), "attitude_quaternion", "quaternion", "body_to_inertial", 4),
        (Lidar(range_noise_std=0.0, los_noise_std=0.0), (np.array([10.0, 0.0, 0.0]),), "range_los", ("m", "unit_vector"), "body", 4),
        (GNSSReceiver(pos_noise_std=0.0, vel_noise_std=0.0), (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])), "position_velocity", ("m", "m/s"), "state_frame", 6),
        (IMU(gyro_params={"noise_std": 0.0}, accel_params={"noise_std": 0.0}), (np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 9.81])), "imu", ("rad/s", "m/s^2"), "body", 6),
    ]

    for sensor, args, quantity, units, frame, size in sensors_and_inputs:
        packet = sensor.measure(*args)
        assert packet.sensor_name == sensor.name
        assert packet.quantity == quantity
        assert packet.units == units
        assert packet.frame == frame
        assert packet.value.shape == (size,)
        assert packet.covariance.shape[0] == packet.covariance.shape[1]
        assert packet.noise_model.model == "gaussian"


def test_load_sensor_suite_from_json_and_yaml(tmp_path: Path):
    json_cfg = tmp_path / "sensors.json"
    json_cfg.write_text(json.dumps({"sensors": [{"type": "gyroscope", "noise_std": 0.0}, {"type": "sun_sensor", "noise_std": 0.0}]}))

    sensors = load_sensor_suite(json_cfg)
    assert [sensor.quantity for sensor in sensors] == ["angular_rate", "sun_vector"]

    yaml_cfg = tmp_path / "sensors.yaml"
    yaml_cfg.write_text("""sensors:
  - type: magnetometer
    noise_std: 0.0
  - type: star_tracker
    noise_std: 0.0
""")
    sensors_yaml = load_sensor_suite(yaml_cfg)
    assert [sensor.quantity for sensor in sensors_yaml] == ["magnetic_field", "attitude_quaternion"]


def test_attitude_sensor_fusion_converges_with_vector_and_quaternion_updates():
    np.random.seed(7)
    dt = 0.1
    q_true = np.array([0.0, 0.0, 0.0, 1.0])
    omega_true = np.array([0.0, 0.0, 0.0])

    fusion = AttitudeSensorFusion(backend="mekf")
    fusion.filter.P = np.eye(6) * 0.5
    fusion.filter.Q = np.eye(6) * 1e-6

    gyro = Gyroscope(noise_std=1e-5, bias_stability=0.0)
    sun_sensor = SunSensor(noise_std=2e-3)
    star_tracker = StarTracker(noise_std=5e-4)
    z_ref_sun = np.array([1.0, 0.0, 0.0])

    for _ in range(60):
        q_true = quat_normalize(quat_mult(q_true, axis_angle_to_quat(omega_true * dt)))
        gyro_packet = gyro.measure(omega_true)
        fusion.predict(gyro_packet)

        sun_body = quat_rot(quat_conj(q_true), z_ref_sun)
        sun_packet = sun_sensor.measure(sun_body)
        sun_packet.metadata["reference"] = z_ref_sun
        fusion.update_from_measurement(sun_packet)

        q_packet = star_tracker.measure(q_true)
        fusion.update_from_measurement(q_packet)

    attitude_error = 1.0 - abs(np.dot(fusion.quaternion, q_true))
    assert attitude_error < 1e-5
    assert np.linalg.norm(fusion.bias) < 1e-3
