"""
Autonomous FDIR Executive and GNC Mission Health State Machine.

Coordinates sensor fault isolation, actuator control reconfiguration, and
autonomous transitions between Nominal, Degraded, Underactuated, and Safe Modes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from opengnc.fdir.actuator_fdir import ActuatorHealth
from opengnc.fdir.sensor_fdir import SensorHealth


class FDIRMode(Enum):
    """Spacecraft GNC operational modes governed by FDIR."""

    NOMINAL = "NOMINAL"
    DEGRADED_POINTING = "DEGRADED_POINTING"
    UNDERACTUATED_CONTROL = "UNDERACTUATED_CONTROL"
    SAFE_MODE_SUN_POINTING = "SAFE_MODE_SUN_POINTING"


class FDIRExecutive:
    """
    Finite State Machine governing autonomous Spacecraft FDIR transitions.

    Evaluates collective sensor health, actuator integrity, and dynamic pointing
    metrics to deterministically transition between operational modes:
    - NOMINAL: Full sensor fusion and nominal 3-axis control.
    - DEGRADED_POINTING: Maintained 3-axis pointing with isolated sensor/actuator.
    - UNDERACTUATED_CONTROL: 2-RW stabilization with auxiliary magnetic damping.
    - SAFE_MODE_SUN_POINTING: Power preservation & tumbling damping fallback.

    Parameters
    ----------
    initial_mode : FDIRMode, optional
        Starting mode for the executive. Default is FDIRMode.NOMINAL.
    max_rate_runaway : float, optional
        Maximum allowable angular velocity (rad/s) before forcing Safe Mode.
        Default is 0.1 rad/s (~5.7 deg/s).
    max_pointing_error_deg : float, optional
        Maximum allowable pointing error (deg) before flagging attitude loss.
        Default is 45.0 degrees.
    """

    def __init__(
        self,
        initial_mode: FDIRMode = FDIRMode.NOMINAL,
        max_rate_runaway: float = 0.1,
        max_pointing_error_deg: float = 45.0,
    ) -> None:
        self.current_mode: FDIRMode = initial_mode
        self.max_rate = float(max_rate_runaway)
        self.max_pointing_err = float(max_pointing_error_deg)

        self.mode_history: list[tuple[float, FDIRMode, str]] = []
        self.transition_log: list[str] = []

    def evaluate_state(
        self,
        time_s: float,
        sensor_health: dict[str, SensorHealth],
        actuator_health: dict[str, ActuatorHealth],
        pointing_error_deg: float = 0.0,
        rate_norm: float = 0.0,
    ) -> FDIRMode:
        """
        Evaluate system health telemetry and perform mode transitions.

        Parameters
        ----------
        time_s : float
            Simulation/mission elapsed time (s).
        sensor_health : dict[str, SensorHealth]
            Dictionary of sensor states {sensor_name: SensorHealth}.
        actuator_health : dict[str, ActuatorHealth]
            Dictionary of actuator states {actuator_name: ActuatorHealth}.
        pointing_error_deg : float, optional
            Current pointing error in degrees. Default is 0.0.
        rate_norm : float, optional
            Current angular rate magnitude ||omega|| in rad/s. Default is 0.0.

        Returns
        -------
        FDIRMode
            Current active FDIR operational mode.
        """
        # Count healthy/functional hardware
        active_sensors = sum(1 for s in sensor_health.values() if s != SensorHealth.ISOLATED)
        active_actuators = sum(1 for a in actuator_health.values() if a != ActuatorHealth.FAILED)

        prev_mode = self.current_mode
        reason = ""

        # Safe Mode Trigger 1: Rate Runaway or severe pointing divergence
        if rate_norm > self.max_rate:
            target_mode = FDIRMode.SAFE_MODE_SUN_POINTING
            reason = f"Angular rate runaway: {rate_norm:.4f} rad/s > {self.max_rate:.4f}"
        # Safe Mode Trigger 2: Insufficient sensors to determine attitude
        elif active_sensors == 0:
            target_mode = FDIRMode.SAFE_MODE_SUN_POINTING
            reason = "Complete loss of attitude sensors (0 active)"
        # Safe Mode Trigger 3: Fewer than 2 functional actuators
        elif active_actuators < 2:
            target_mode = FDIRMode.SAFE_MODE_SUN_POINTING
            reason = f"Catastrophic actuator failure: {active_actuators} functional actuators"
        # Underactuated Control: Exactly 2 functional actuators
        elif active_actuators == 2:
            target_mode = FDIRMode.UNDERACTUATED_CONTROL
            reason = "Actuator count reduced to 2: initiating underactuated control"
        # Degraded Pointing: Any isolated sensor or degraded/failed actuator (with >= 3 remaining)
        elif (
            any(s == SensorHealth.ISOLATED for s in sensor_health.values())
            or any(a != ActuatorHealth.HEALTHY for a in actuator_health.values())
            or pointing_error_deg > self.max_pointing_err
        ):
            target_mode = FDIRMode.DEGRADED_POINTING
            reason = "Subsystem degraded; 3-axis pointing maintained via reallocation"
        else:
            target_mode = FDIRMode.NOMINAL
            reason = "All subsystems nominal"

        # Apply transition
        if target_mode != prev_mode:
            msg = f"[{time_s:.2f}s] FDIR Transition: {prev_mode.value} -> {target_mode.value} ({reason})"
            self.transition_log.append(msg)
            self.mode_history.append((time_s, target_mode, reason))
            self.current_mode = target_mode

        return self.current_mode

    def force_safe_mode(self, time_s: float, reason: str = "Manual override") -> None:
        """Manually force executive into Safe Mode."""
        prev = self.current_mode
        self.current_mode = FDIRMode.SAFE_MODE_SUN_POINTING
        msg = (
            f"[{time_s:.2f}s] FDIR Transition: {prev.value} -> {self.current_mode.value} ({reason})"
        )
        self.transition_log.append(msg)
        self.mode_history.append((time_s, self.current_mode, reason))

    def reset_to_nominal(self, time_s: float) -> None:
        """Reset executive back to Nominal Mode after recovery."""
        prev = self.current_mode
        self.current_mode = FDIRMode.NOMINAL
        msg = f"[{time_s:.2f}s] FDIR Transition: {prev.value} -> {self.current_mode.value} (Recovery reset)"
        self.transition_log.append(msg)
        self.mode_history.append((time_s, self.current_mode, "Recovery reset"))

    def get_telemetry(self) -> dict[str, Any]:
        """
        Generate diagnostic telemetry summary of FDIR status.

        Returns
        -------
        dict[str, Any]
            Dictionary containing current mode, transition history, and log.
        """
        return {
            "current_mode": self.current_mode.value,
            "transition_count": len(self.mode_history),
            "transition_log": list(self.transition_log),
        }
