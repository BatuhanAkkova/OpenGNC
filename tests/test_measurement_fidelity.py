import numpy as np
import pytest

from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.sensor import Sensor


class MockSensor(Sensor):
    def measure(self, true_state, **kwargs):
        return self.build_measurement(true_state)


def test_fogm_noise():
    sensor = MockSensor()
    sigma = 1.0
    tau = 10.0
    dt = 0.1

    current_val = 0.0
    values = []
    for _ in range(100):
        current_val = sensor.apply_fogm_noise(current_val, sigma, tau, dt)
        values.append(current_val)

    assert np.any(np.abs(values) > 0)
    diffs = np.diff(values)
    assert np.std(diffs) < sigma


def test_calibration():
    sensor = MockSensor()
    val = np.array([1.0, 0.0, 0.0])

    cal = sensor.apply_calibration(val, scale_factor=1.1)
    assert np.allclose(cal, np.array([1.1, 0, 0]))

    cal = sensor.apply_calibration(val, bias=np.array([0.1, 0, 0]))
    assert np.allclose(cal, np.array([1.1, 0, 0]))

    mis = np.array([[0, 0.1, 0], [0, 0, 0], [0, 0, 0]])
    cal = sensor.apply_calibration(val, misalignment=mis)
    assert np.allclose(cal, np.array([1.0, 0, 0]))

    mis2 = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    cal = sensor.apply_calibration(val, misalignment=mis2)
    assert np.allclose(cal, np.array([1.0, 0.1, 0.0]))


def test_fault_injection():
    sensor = MockSensor()
    val = np.array([1.0, 1.0, 1.0])

    sensor.fault_state = "stuck"
    sensor.stuck_value = np.array([0.0, 0.0, 0.0])
    assert np.allclose(sensor.apply_faults(val), np.zeros(3))

    sensor.fault_state = None
    assert np.allclose(sensor.apply_faults(val), val)

    sensor.fault_state = "spike"
    spike_val = sensor.apply_faults(val)
    assert not np.allclose(spike_val, val)


def test_gyroscope_fidelity():
    mis = np.zeros((3, 3))
    mis[1, 0] = 0.1
    gyro = Gyroscope(noise_std=0.0, bias_stability=0.0, misalignment=mis, scale_factor=1.1)

    w_true = np.array([1.0, 0.0, 0.0])
    w_meas = gyro.measure(w_true).value

    assert np.allclose(w_meas, np.array([1.1, 0.11, 0.0]))
