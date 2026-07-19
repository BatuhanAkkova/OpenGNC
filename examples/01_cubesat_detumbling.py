"""
CubeSat Detumbling Example
==========================

This script demonstrates a simple Attitude Determination and Control System (ADCS) simulation
for a CubeSat.

Scenario:
    - 3U CubeSat deployed with high initial tumble rates (10 deg/s per axis).
    - Actuators: 3-axis Magnetorquers.
    - Sensors: 3-axis Magnetometer.
    - Control: B-Dot algorithm for detumbling.
    - Environment: 
        - Magnetic Field: Tilted Dipole (Calculated in ECEF, rotated to ECI).
        - Disturbance: Gravity Gradient Torque.

Dynamics:
    - Rigid Body Euler Equations with Environment Torques.
    - Quaternion Kinematics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime

# Add src to path to ensure we can import opengnc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.attitude_dynamics.rigid_body import euler_equations
from opengnc.environment.mag_field import tilted_dipole_field
from opengnc.disturbances.gravity import GradientTorque
from opengnc.sensors.magnetometer import Magnetometer
from opengnc.actuators.magnetorquer import Magnetorquer
from opengnc.classical_control.bdot import BDot
from opengnc.integrators.rk4 import RK4
from opengnc.utils.quat_utils import quat_rot, quat_normalize, quat_conj, quat_to_rmat
from opengnc.utils.frame_conversion import eci2ecef, ecef2eci
from opengnc.utils.time_utils import calc_jd

def simulation():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    
    # Simulation Parameters
    dt = 0.1 # Time step [s]
    t_max = 5000.0 # Duration [s] (Approx 1.5 hours)
    chk_rate = 1.0 # Data recording rate [s]
    
    # Satellite Properties (3U CubeSat approx)
    # J aligned with body axes
    J = np.diag([0.025, 0.025, 0.005]) 
    
    # Initial State
    state = np.zeros(7)
    state[3] = 1.0 # Quaternion Identity [0,0,0,1]
    # Initial Tumble
    state[4:7] = np.deg2rad([10.0, -10.0, 5.0])
    
    t = 0.0
    
    # Orbit Parameters (Circular, Polar, 500 km)
    Re = 6371.0 * 1000 # Earth Radius [m]
    h = 500.0 * 1000 # Altitude [m]
    mu = 3.986e14 # Earth GM [m^3/s^2]
    r_orbit = Re + h
    period = 2 * np.pi * np.sqrt(r_orbit**3 / mu)
    n = 2 * np.pi / period # Mean motion
    
    # Epoch for Frame Conversion
    epoch = datetime.datetime(2025, 1, 1, 12, 0, 0)
    
    print(f"Orbit Period: {period/60:.2f} mins")
    
    # Components
    mag_sensor = Magnetometer(noise_std=50e-9) 
    mtq = Magnetorquer(max_dipole=0.2)         
    bdot_ctrl = BDot(gain=100000.0)            
    integrator = RK4()
    gg = GradientTorque()
    
    # -------------------------------------------------------------------------
    # 2. Dynamics Function
    # -------------------------------------------------------------------------
    def dynamics(t_curr, y_curr, torque_ext_body):
        q = y_curr[0:4]
        w = y_curr[4:7]
        q = q / np.linalg.norm(q)
        
        # Quaternion Kinematics: q_dot = 0.5 * q x [w; 0]
        wx, wy, wz = w
        Omega = np.array([
            [0, wz, -wy, wx],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])
        q_dot = 0.5 * Omega @ q
        
        # Rigid Body Dynamics
        w_dot = euler_equations(J, w, torque_ext_body)
        
        return np.concatenate((q_dot, w_dot))

    # -------------------------------------------------------------------------
    # 3. Simulation Loop
    # -------------------------------------------------------------------------
    data_t = []
    data_w = []
    data_m = []
    data_egg = []
    
    b_body_prev = None
    
    print("Starting Simulation...")
    while t < t_max:
        # Environment (Orbit)
        # Propagate Orbit in ECI (Polar)
        theta = n * t
        r_eci = r_orbit * np.array([np.cos(theta), 0, np.sin(theta)])
        
        # Calculate Julian Date
        jd_day, jd_frac = calc_jd(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + t)
        jd = jd_day + jd_frac
        
        # Magnetic Field (ECEF -> ECI)
        r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd)
        b_ecef = tilted_dipole_field(r_ecef)
        b_eci_true, _ = ecef2eci(b_ecef, np.zeros(3), jd)
        
        # Sensors & Actuators
        q_curr = state[0:4]
        q_inv = quat_conj(q_curr) # ECI -> Body
        
        b_body_true = quat_rot(q_inv, b_eci_true)
        b_body_meas = mag_sensor.measure(b_body_true).value
        
        # Control
        if b_body_prev is None:
            b_dot_est = np.zeros(3)
        else:
            b_dot_est = (b_body_meas - b_body_prev) / dt
            
        m_cmd = bdot_ctrl.calculate_control(b_dot_est)
        m_actual = mtq.command(m_cmd)
        b_body_prev = b_body_meas
        
        # Disturbance Torques
        # Gravity Gradient
        t_gg = gg.gravity_gradient_torque(J, r_eci, q_curr)
        
        # Total Torque
        t_ctrl = np.cross(m_actual, b_body_true)
        t_total = t_ctrl + t_gg
        
        # Propagate
        state, t, _ = integrator.step(dynamics, t, state, dt, torque_ext_body=t_total)
        state[0:4] = quat_normalize(state[0:4])
        
        # Record
        if t % chk_rate < dt:
            data_t.append(t)
            data_w.append(state[4:7])
            data_m.append(m_actual)
            data_egg.append(t_gg)
            
            if len(data_t) % 100 == 0:
                 w_mag = np.linalg.norm(state[4:7]) * 180/np.pi
                 print(f"Time: {t:.1f}s | Ang Vel Mag: {w_mag:.2f} deg/s")

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------
    data_t = np.array(data_t)
    data_w = np.array(data_w) * 180.0 / np.pi
    data_m = np.array(data_m)
    data_egg = np.array(data_egg)
    
    print("Simulation Complete.")
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    ax[0].plot(data_t, data_w)
    ax[0].set_ylabel('Ang Vel [deg/s]')
    ax[0].set_title('Detumbling with Gravity Gradient')
    ax[0].grid(True)
    ax[0].legend(['x','y','z'])
    
    ax[1].plot(data_t, data_m)
    ax[1].set_ylabel('Dipole [Am^2]')
    ax[1].grid(True)
    
    ax[2].plot(data_t, data_egg * 1e6) # Scale to uNm
    ax[2].set_ylabel('Gravity Gradient [uNm]')
    ax[2].set_xlabel('Time [s]')
    ax[2].grid(True)
    
    plt.tight_layout()
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'detumbling.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    simulation()




