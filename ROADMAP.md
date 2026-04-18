# OpenGNC — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — bridging the gap between academic simulation and flight-ready software across the full mission lifecycle.

This roadmap outlines the strategic development of **OpenGNC** to provide a high-fidelity, validated, and performance-optimized platform for modern aerospace missions.

---

## Strategic Objectives

1.  **High-Fidelity Transparency**: Every algorithm and environment model must be fully documented and traceable to published aerospace standards.
2.  **Mission Lifecycle Coverage**: Support a seamless path from fundamental research and conceptual design to embedded flight code and mission operations.
3.  **Performance & Portability**: Prioritize C++ acceleration, SIMD-optimized state management, and thread-safe, lock-free communication for high-performance computing.
4.  **Autonomous Intelligence**: Integrate modern AI/ML methods for automated state estimation, decision-making, and fault isolation.

---

## Development Phases

### Phase 1: High-Fidelity Foundation & Design (In Progress)
*Establishing the core orbital mechanics and environmental simulation environment.*
- [x] **Recursive Gravity Models**: Optimized EGM2008 up to degree/order 360+ via Cython/Numba.
- [x] **High-Fidelity Environment**: Integrated IGRF-13 magnetic fields and NRLMSISE-00 atmospheric density models.
- [x] **Advanced Propagators**: Cowell’s Method, Keplerian elements, and Gauss-Variational Equations (GVE).
- [x] **Professionality**: Established CI/CD pipeline, automated formatting, and performance benchmarking suite.
- [x] **Interoperability**: Standardized interfaces for external tools like GMAT, Orekit, and STK (SPICE kernels).

### Phase 2: Autonomy, AI & Advanced Research (Later)
*Integrating modern control and state estimation techniques for autonomous missions.*
- [ ] **Relative Navigation**: Filter-based relative state estimation for swarming, docking, and proximity operations.
- [ ] **Optimal Control**: Expanded MPC optimization using CasADi for highly constrained multi-burn maneuvers.
- [ ] **AI/ML Hooks**: Reinforcement Learning (RL) wrappers for autonomous rendezvous and docking (GNC Gymnasium).
- [ ] **FDIR v2**: Implementation of PCA and LSTM-based anomaly detection for sensor/actuator protection.

### Phase 3: Flight Implementation & Embedded Software
*Converting high-fidelity simulations into deterministic, flight-ready artifacts.*
- [x] **C++ Acceleration**: Porting core filters (MEKF, UKF) and control laws to header-only, C++17/20 templates.
- [x] **Hardware-in-the-Loop (HIL)**: Generic SIM-to-Hardware bridge with Serial and UDP (Packet Framing & CRC).
- [x] **Real-Time Performance**: Enforce static memory allocation and lock-free message passing for mission-critical paths.
- [x] **Verification**: Automated Monte Carlo analysis suite for robust stability and performance margin proofs.

### Phase 4: Mission Operations & Space Situational Awareness
*Supporting the platform in active operations and orbital safety.*
- [x] **Ground Station Ops**: CCSDS packet parsing (Primary Header + CUC), de-commutation toolkits, and Real-time Plotting Dashboards.
- [x] **SSA/SDA**: Conjunction Assessment (CAT), TLE maintenance, and automated collision avoidance planning (CAM Optimization).
- [x] **Mission Control API**: Specialized FastAPI/WebSocket interface for live telemetry monitoring and commanding.

---

*Last updated: March 2026 | Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)*
