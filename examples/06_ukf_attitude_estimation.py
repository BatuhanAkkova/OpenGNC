"""
Attitude Estimation with Unscented Kalman Filter (UKF)
======================================================

This example demonstrates how to use the UKF for satellite attitude estimation
by fusing a star tracker and a gyroscope.

Scenario:
    - 3-axis gyro (rate measurements with bias and noise).
    - Star tracker (quaternion measurements).
    - UKF handles the quaternion state on a manifold using tangent space errors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.kalman_filters.ukf import UKF_Attitude
from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.star_tracker import StarTracker
from opengnc.utils.quat_utils import quat_mult, quat_normalize, axis_angle_to_quat, quat_conj

def run_example():
    # Configuration
    dt = 0.1
    t_max = 100.0
    time = np.arange(0, t_max, dt)
    
    # Sensors
    gyro = Gyroscope(initial_bias=np.array([0.01, -0.01, 0.005]), noise_std=0.001)
    st = StarTracker(noise_std=0.0001) # Very accurate
    
    # Filter Initialization
    ukf = UKF_Attitude(alpha=1e-3)
    ukf.P *= 0.1
    ukf.Q = np.eye(6) * 1e-6
    ukf.R = np.eye(3) * 1e-8 # Star tracker is very accurate
    
    # Truth state
    q_true = np.array([0, 0, 0, 1.0])
    omega_true = np.array([0.05, 0.02, -0.01]) # Constant rate
    
    # Simulation Loop
    results_t = []
    results_q_err = []
    results_bias_err = []
    
    print("Running UKF Attitude Estimation...")
    for t in time:
        # Step Truth
        dq_true = axis_angle_to_quat(omega_true * dt)
        q_true = quat_normalize(quat_mult(q_true, dq_true))
        
        # Measurements
        gyro_packet = gyro.measure(omega_true, dt=dt)

        # Update ST at 1Hz
        st_available = (int(t/dt) % int(1.0/dt) == 0)

        # Predict
        ukf.predict(gyro_packet)

        # Update
        if st_available:
            star_tracker_packet = st.measure(q_true)
            ukf.update(star_tracker_packet)
            
        # Logging
        # Quaternion error (angle)
        q_err = quat_mult(quat_conj(ukf.x[:4]), q_true)
        angle_err = 2 * np.linalg.norm(q_err[:3]) * 180/np.pi
        
        bias_err = np.linalg.norm(ukf.x[4:] - gyro.current_bias)
        
        results_t.append(t)
        results_q_err.append(angle_err)
        results_bias_err.append(bias_err)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results_t, results_q_err)
    plt.ylabel('Attitude Error [deg]')
    plt.title('UKF Attitude Estimation Performance')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_t, results_bias_err)
    plt.ylabel('Bias Error [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    
    plt.tight_layout()
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'ukf_attitude.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_example()




