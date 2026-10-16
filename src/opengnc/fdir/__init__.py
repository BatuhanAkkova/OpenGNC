"""
OpenGNC Fault Detection, Isolation, and Recovery (FDIR) Subsystem.

Provides autonomous sensor fault detection, statistical innovation testing,
actuator fault-tolerant control reallocation, underactuated attitude stabilization,
and mission-level operational health management.
"""

from __future__ import annotations

from opengnc.fdir.actuator_fdir import (
    ActuatorFaultDetector,
    ActuatorHealth,
    UnderactuatedAttitudeController,
    WeightedControlAllocator,
)
from opengnc.fdir.executive import (
    FDIRExecutive,
    FDIRMode,
)
from opengnc.fdir.sensor_fdir import (
    ChiSquareMonitor,
    GLRTMonitor,
    SensorHealth,
    SensorIsolationManager,
)

__all__ = [
    "ActuatorFaultDetector",
    "ActuatorHealth",
    "ChiSquareMonitor",
    "FDIRMode",
    "FDIRExecutive",
    "GLRTMonitor",
    "SensorHealth",
    "SensorIsolationManager",
    "UnderactuatedAttitudeController",
    "WeightedControlAllocator",
]
