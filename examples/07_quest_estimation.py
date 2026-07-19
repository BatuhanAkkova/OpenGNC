"""
Deterministic Attitude Determination with QUEST Algorithm
==========================================================

This example demonstrates how to determine the attitude of a spacecraft
using the QUEST (Quaternion Estimator) algorithm from two or more vector 
measurements.

Scenario:
    - Sensors: Magnetometer and Sun Sensor providing body-frame vectors.
    - Reference: IGRF Magnetic Field and Analytical Sun Position in ECI.
    - Algorithm: QUEST for optimal quaternion (Inertial -> Body).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.attitude_determination.quest import quest
from opengnc.utils.quat_utils import quat_rot, quat_conj
from opengnc.utils.state_conversion import quat_to_euler
from opengnc.sensors.magnetometer import Magnetometer
from opengnc.sensors.sun_sensor import SunSensor

def run_example():
    # Truth Attitude (Random)
    # Roll=10, Pitch=20, Yaw=30 deg
    euler_true = np.radians([10, 20, 30])
    # For simplicity, let's just define a rotation matrix or quaternion
    from opengnc.utils.state_conversion import euler_to_quat
    q_true = euler_to_quat(euler_true, sequence="321")
    
    # Reference Vectors (Inertial Frame)
    # e.g., Sun and Magnetic field
    v_sun_eci = np.array([1.0, 0.0, 0.0])
    v_mag_eci = np.array([0.0, 0.0, 1.0])
    
    ref_vectors = [v_sun_eci, v_mag_eci]
    
    # Simulate Measurements (Body Frame)
    mag_sensor = Magnetometer(noise_std=0.001)
    sun_sensor = SunSensor(noise_std=0.001)
    
    q_inv = quat_conj(q_true) # ECI -> Body
    
    v_sun_body_true = quat_rot(q_inv, v_sun_eci)
    v_mag_body_true = quat_rot(q_inv, v_mag_eci)
    
    v_sun_meas = sun_sensor.measure(v_sun_body_true).value
    v_mag_meas = mag_sensor.measure(v_mag_body_true).value
    
    body_vectors = [v_sun_meas, v_mag_meas]
    
    # Weights (Star tracker / Sun sensor usually more weighted than Mag)
    weights = [0.8, 0.2]
    
    # Run QUEST
    print("Running QUEST Algorithm...")
    q_est = quest(body_vectors, ref_vectors, weights=weights)
    
    # Results
    euler_est = quat_to_euler(q_est, sequence="321")
    
    print("-" * 30)
    print(f"True Euler [deg]:  {np.degrees(euler_true)}")
    print(f"Est. Euler [deg]:  {np.degrees(euler_est)}")
    print("-" * 30)
    
    error_angle = np.degrees(np.linalg.norm(euler_true - euler_est))
    print(f"Total Angular Error: {error_angle:.4f} degrees")
    print("-" * 30)

    # Visualization
    labels = ['Roll', 'Pitch', 'Yaw']
    true_deg = np.degrees(euler_true)
    est_deg = np.degrees(euler_est)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, true_deg, width, label='True Orientation')
    plt.bar(x + width/2, est_deg, width, label='QUEST Estimated')
    
    plt.ylabel('Angle [deg]')
    plt.title('QUEST Attitude Determination Performance')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'quest_results.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_example()




