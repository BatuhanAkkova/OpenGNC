"""
Attitude Estimation (MEKF) Example
==================================

This script demonstrates sensor fusion for attitude determination using a 
Multiplicative Extended Kalman Filter (MEKF).

Scenario:
    - A spacecraft is tumbling freely.
    - Sensors:
        1. Gyroscope (High Rate, Biased, Noisy).
        2. Star Tracker (Low Rate, Quaternion Measurements, Noisy).
    - Filter estimates proper attitude (Quaternion) and Gyro Bias.

Dynamics:
    - Rigid Body Euler Equations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.kalman_filters.mekf import MEKF
from opengnc.attitude_dynamics.rigid_body import euler_equations
from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.star_tracker import StarTracker
from opengnc.utils.quat_utils import quat_mult, quat_conj, quat_normalize, axis_angle_to_quat

def simulation():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    
    # Time Settings
    dt_sim = 0.01 # 100 Hz Dynamics/Gyro
    dt_st = 1.0 # 1 Hz Star Tracker
    duration = 60.0 # 60s simulation
    
    # True System Properties
    J = np.diag([0.1, 0.12, 0.15])
    
    # Initial State (Truth)
    q_true = np.array([0.0, 0.0, 0.0, 1.0])
    w_true = np.deg2rad(np.array([5.0, -2.0, 1.0])) # Slow tumble
    
    # Sensor Properties
    # Gyro
    gyro_bias_true = np.deg2rad(np.array([1.0, -1.0, 0.5])) # Large bias (1 deg/s)
    gyro_noise_std = np.deg2rad(0.1)  # 0.1 deg/s noise
    
    # Star Tracker (Quaternion packets)
    st_noise_std = 0.005
    
    # Sensor Initialization
    gyro = Gyroscope(noise_std=gyro_noise_std, initial_bias=gyro_bias_true, dt=dt_sim)
    st = StarTracker(noise_std=st_noise_std)
    
    # Filter Initialization
    mekf = MEKF(q_init=[0.0, 0.0, 0.0, 1.0], beta_init=[0.0, 0.0, 0.0])
    
    # Tuning
    mekf.P = np.eye(6) * 1.0 # High initial uncertainty
    mekf.Q = np.eye(6) * 1e-6 # Process noise
    mekf.Q[3:6, 3:6] *= 1e-4 # Bias random walk
    mekf.R = np.eye(3) * (st_noise_std**2)
    
    # -------------------------------------------------------------------------
    # 2. Simulation Loop
    # -------------------------------------------------------------------------
    t = 0.0
    
    history = {
        't': [],
        'att_err': [],
        'bias_est': [],
        'bias_true': []
    }
    
    print("Starting MEKF Simulation...")
    print(f"True Bias: {np.rad2deg(gyro_bias_true)} deg/s")
    
    while t < duration:
        # Propagate Truth
        dw = euler_equations(J, w_true, np.zeros(3)) * dt_sim
        w_true += dw
        
        angle = np.linalg.norm(w_true) * dt_sim
        axis = w_true / np.linalg.norm(w_true) if angle > 0 else np.array([1,0,0])
        curr_dq = axis_angle_to_quat(axis * angle)
        q_true = quat_normalize(quat_mult(q_true, curr_dq))
        
        # Gyro Measurement
        gyro_packet = gyro.measure(w_true, dt=dt_sim)

        # Filter Prediction (High Rate)
        mekf.predict(gyro_packet)

        # Filter Update (Low Rate)
        if t > 0 and (t % dt_st < dt_sim):
            star_tracker_packet = st.measure(q_true)
            mekf.update(star_tracker_packet)

        # Analysis
        # Attitude Error (Angle between q_est and q_true)
        # q_err = q_est * q_true_inv
        q_err = quat_mult(mekf.q, quat_conj(q_true))
        # Angle = 2 * acos(w)
        theta_err = 2 * np.arccos(np.clip(abs(q_err[3]), -1, 1))
        theta_err_deg = np.rad2deg(theta_err)
        
        history['t'].append(t)
        history['att_err'].append(theta_err_deg)
        history['bias_est'].append(mekf.beta.copy()) # Copy to avoid ref issues
        history['bias_true'].append(gyro_bias_true.copy())
        
        t += dt_sim
        
        if t % 10.0 < dt_sim:
            print(f"Time: {t:.1f}s | Att Err: {theta_err_deg:.4f} deg | Bias Est X: {np.rad2deg(mekf.beta[0]):.4f}")

    # -------------------------------------------------------------------------
    # 3. Visualization
    # -------------------------------------------------------------------------
    t_arr = np.array(history['t'])
    err_arr = np.array(history['att_err'])
    bias_est = np.array(history['bias_est']) * 180/np.pi
    bias_true = np.array(history['bias_true']) * 180/np.pi
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Attitude Error
    ax[0].plot(t_arr, err_arr, 'r', label='Error')
    ax[0].set_ylabel('Attitude Error [deg]')
    ax[0].set_title('MEKF Attitude Estimation Performance')
    ax[0].set_yscale('log')
    ax[0].grid(True, which='both')
    ax[0].legend()
    
    # Bias Estimation
    ax[1].plot(t_arr, bias_est[:, 0], 'r', label='Est Bx')
    ax[1].plot(t_arr, bias_est[:, 1], 'g', label='Est By')
    ax[1].plot(t_arr, bias_est[:, 2], 'b', label='Est Bz')
    ax[1].set_prop_cycle(None) # Reset colors to plot dashed lines matching
    ax[1].plot(t_arr, bias_true[:, 0], 'r--', alpha=0.5, label='True Bx')
    ax[1].plot(t_arr, bias_true[:, 1], 'g--', alpha=0.5, label='True By')
    ax[1].plot(t_arr, bias_true[:, 2], 'b--', alpha=0.5, label='True Bz')
    ax[1].set_ylabel('Gyro Bias [deg/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].grid(True)
    ax[1].legend(loc='right')
    
    plt.tight_layout()
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'attitude_est.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    simulation()




