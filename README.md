# OpenGNC

[![PyPI version](https://img.shields.io/pypi/v/OpenGNC.svg)](https://pypi.org/project/OpenGNC/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**OpenGNC** is a professional-grade, high-fidelity Guidance, Navigation, and Control (GNC) library for spacecraft simulation, orbital mechanics, and mission analysis. Built with SI units and aerospace standards, it provides a comprehensive suite of tools for satellite engineers and researchers.

**[Explore the Documentation](https://BatuhanAkkova.github.io/OpenGNC/)** | **[View Examples](examples/)** | **[Contributing](CONTRIBUTING.md)**

---

## Key Highlights

- **SI Unit Compliance**: All calculations strictly use SI units (meters, kilograms, seconds, radians) unless otherwise explicitly suffixed.
- **High-Fidelity Environment**: Integrated models for NRLMSISE-00 density, IGRF-13 magnetic fields, and EGM2008 gravity harmonics.
- **Validated Algorithms**: Implementations of industry-standard filters (MEKF, UKF) and deterministic methods (QUEST, TRIAD).
- **Optimization Ready**: Built-in support for Optimal Control (LQR) and Model Predictive Control (MPC) using CasADi.
- **Modular Architecture**: Easily swap integrators, disturbance models, or sensors for customized simulations.

---

## Extensive Features

### Environment & Space Weather
- **Atmospheric Models**: NRLMSISE-00 (high-fidelity), Harris-Priester (diurnal bulge), and Exponential density.
- **Geomagnetic Models**: IGRF-13 (International Geomagnetic Reference Field) and Tilted Dipole models.
- **Solar & Ephemeris**: Analytical solar position, shadow models (Umbra/Penumbra), and Planetary ephemeris.
- **Gravity Field**: Recursive EGM2008 Spherical Harmonics (up to N-degree), J2, and Two-body attraction.

### Orbital Mechanics & Propagators
- **Propagators**: Cowell’s Method, Keplerian elements, and SGP4 for TLE propagation.
- **Numerical Integrators**: High-order fixed and adaptive steps: RK4, RK45 (Runge-Kutta-Fehlberg), and DOP853.
- **Orbital Maneuvers**: Hohmann transfer, Bi-elliptic, Phasing, Plane changes, and Lambert Targeting.
- **Initial Orbit Determination (IOD)**: Robust Gauss, Laplace, and universal variable methods.

### Attitude Dynamics & Determination
- **Kinematics**: Quaternion, Euler Angle, and DCM (Direction Cosine Matrix) transformations.
- **Dynamics**: Euler's equations for rigid body rotation, including variable inertia and disturbances.
- **Determination**: Deterministic TRIAD and QUEST algortihms for attitude from vector observations.

### Guidance, Navigation & Control (GNC)
- **State Estimation**: 
  - **MEKF**: Multiplicative Extended Kalman Filter for attitude.
  - **UKF**: Unscented Kalman Filter for non-linear state estimation.
  - **EKF/KF**: Standard and Extended Kalman Filters for orbital state mapping.
- **Control Law Design**:
  - **Classic**: PID controllers and B-Dot detumbling logic.
  - **Optimal**: LQR (Linear Quadratic Regulator) and LQE (Kalman Filter design).
  - **Advanced**: Nonlinear MPC (Model Predictive Control) and Sliding Mode Control.
- **Sensors**: Realistic Star Tracker, Sun Sensor, Magnetometer, Gyroscope, IMU, GNSS, and ranging sensors with a shared measurement packet interface.
- **Actuators**: Model Reaction Wheels (saturation/jitter) and Thrusters (Chemical/Electric).

### Mission Design
- **SSA (Space Situational Awareness)**: Conjunction Assessment (CAT), Maneuver Detection, and TLE interface.
- **Mission Design**: $\Delta v$ Budgeting, Communication Link Budgets, and Ground Station Coverage tools.

### Visualization & Analysis
- **3D Trajectories**: Interactive Plotly-based orbital trajectory and attitude visualization.
- **Coordinate Frames**: Visualizers for ECI, ECEF, Hill (RSW), and Body frame transformations.

### Real-Time Performance & Determinism
- **Static Allocation**: Core Kalman Filters (MEKF, UKF) are implemented in header-only C++17/20 with strict static memory allocation (no `malloc`/`new` in hot paths).
- **SIMD Acceleration**: Full Eigen integration for vectorized matrix operations on ARM and x86 architectures.
- **Lock-Free Communication**: High-speed, wait-free **SPSC (Single-Producer Single-Consumer) Queue** for telemetry and command dispatching.

### Mission Verification Suite
- **Monte Carlo Harness**: Scale simulations to 10,000+ runs with parallel processor execution.
- **Statistical Analyzer**: Built-in tools for **3-Sigma margin proofs**, convergence rates, and stability limits.
- **Consistency Verification**: Automated **NIS (Normalized Innovation Squared)** and **NEES** tests to validate filter optimality against stochastic truth.

---

## Installation

### From PyPI (Recommended)
```bash
pip install opengnc
```

#### With MPC support (Optional)
```bash
pip install "opengnc[mpc]"
```

#### With Interoperability Support (Optional)
```bash
pip install "opengnc[interop]"
```

#### With Operational Tools (Optional)
```bash
pip install "opengnc[ops]"
```

### From Source (Development)
```bash
git clone https://github.com/BatuhanAkkova/opengnc.git
cd opengnc
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from opengnc.kalman_filters import AttitudeSensorFusion
from opengnc.sensors import Gyroscope, SunSensor

fusion = AttitudeSensorFusion(backend="mekf")
gyro = Gyroscope(noise_std=5e-4)
sun_sensor = SunSensor(noise_std=1e-3)

gyro_packet = gyro.measure(np.array([0.01, -0.02, 0.015]))
fusion.predict(gyro_packet)

sun_packet = sun_sensor.measure(np.array([1.0, 0.0, 0.0]))
sun_packet.metadata["reference"] = np.array([1.0, 0.0, 0.0])
fusion.update_from_measurement(sun_packet)
```

## Sensor Interface

Every sensor now uses one API: `measure(...)` returns a `SensorMeasurement` object with consistent `quantity`, `units`, `frame`, `covariance`, and `noise_model` fields.

The fusion layer depends only on `SensorMeasurement`, not on any specific concrete sensor class.

## Config-Driven Sensor Suites

Use `opengnc.sensors.load_sensor_suite(...)` with JSON or YAML to select sensors without editing Python source.
An example config is provided at `examples/configs/attitude_sensor_suite.yaml`.

## Optional Namespaces

The default `opengnc` install is centered on spacecraft GNC, simulation, and mission analysis.

- Operational tooling now lives under `opengnc.ops`, including dashboard and ground-segment helpers.
- Experimental modules now live under `opengnc.experimental`, including EDL and GMAT integration.
- Legacy imports remain available for compatibility and emit deprecation warnings.

---

## Example Simulations

| Application | Description | Visualization | Script |
| :--- | :--- | :--- | :--- |
| **CubeSat Detumbling** | B-Dot magnetic control using noisy magnetometer data. | ![CubeSat Detumbling](assets/detumbling.png) | [01_cubesat_detumbling.py](examples/01_cubesat_detumbling.py) |
| **MPC Rendezvous** | Optimal multi-burn approach in GEO using NMPC. | ![Autonomous Rendezvous](assets/rendezvous.png) | [04_autonomous_rendezvous.py](examples/04_autonomous_rendezvous.py) |
| **MEKF Estimation** | High-fidelity orientation tracking fusing star tracker & gyro. | ![Attitude Estimation](assets/attitude_est.png) | [05_attitude_estimation_mekf.py](examples/05_attitude_estimation_mekf.py) |
| **VLEO Maintenance** | Altitude keeping in high-drag orbits using electric propulsion. | ![VLEO Orbit Maintenance](assets/vleo_maintenance.png) | [02_vleo_orbit_maintenance.py](examples/02_vleo_orbit_maintenance.py) |
| **Gauss IOD** | Initial Orbit Determination from 3-LoS vectors. | ![Gauss IOD](assets/gauss_iod_results.png) | [10_gauss_iod_determination.py](examples/10_gauss_iod_determination.py) |
| **Viz & Plots** | Comprehensive orbital and attitude visualization demo. | ![Visualization](assets/orbit_viz.png) | [11_visualization_demo.py](examples/11_visualization_demo.py) |
| **Mission Sim** | Core demonstration of the simulation and logging framework. | - | [example_simulation.py](examples/example_simulation.py) |

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

## Contributing & Support

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. 
For support or feedback, please contact [Batuhan Akkova](mailto:batuhanakkova1@gmail.com).

