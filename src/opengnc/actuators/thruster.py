"""
Thruster models including Chemical, Electric, and Multi-thruster clusters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.actuators.actuator import Actuator


class Thruster(Actuator):
    """
    Base Thruster model.

    Produces thrust force and consumes mass based on Isp.

    Parameters
    ----------
    max_thrust : float, optional
        Maximum thrust (N). Default is 1.0.
    min_impulse_bit : float, optional
        Minimum impulse bit (Ns). Default is 0.0.
    isp : float, optional
        Specific Impulse (s). Default is None.
    name : str, optional
        Actuator name. Default is "Thruster".
    """

    def __init__(
        self,
        max_thrust: float = 1.0,
        min_impulse_bit: float = 0.0,
        isp: float | None = None,
        name: str = "Thruster",
    ) -> None:
        """
        Initialize thruster base.

        Parameters
        ----------
        max_thrust : float, optional
            Maximum thrust (N). Default 1.0.
        min_impulse_bit : float | None, optional
            Minimum impulse bit (Ns). Default 0.0.
        isp : float | None, optional
            Specific Impulse (s).
        name : str, optional
            Actuator name. Default "Thruster".
        """
        super().__init__(name=name, saturation=max_thrust)
        self.max_thrust: float = max_thrust
        self.min_impulse_bit: float = min_impulse_bit
        self.isp: float | None = isp

    def command(
        self, thrust_cmd: float | None = None, dt: float | None = None, *args: Any, **kwargs: Any
    ) -> float:
        """
        Calculate delivered thrust.

        Parameters
        ----------
        thrust_cmd : float
            Commanded thrust (N).
        dt : float, optional
            Time step duration (s). Required for MIB checks.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            Delivered thrust (N).
        """
        # Saturation (clip to max thrust)
        if thrust_cmd is None:
            if not args:
                raise ValueError("thrust_cmd is required.")
            thrust_cmd = float(args[0])
        thrust = float(self.apply_saturation(thrust_cmd))

        # Minimum Impulse Bit Logic
        # If dt is provided, check if the requested impulse is possible.
        if dt is not None and self.min_impulse_bit > 0 and abs(thrust) > 1e-9:
            requested_impulse: float = abs(thrust) * dt
            if requested_impulse < self.min_impulse_bit:
                # Deadband behavior for impulses < MIB
                thrust = 0.0

        return float(thrust)

    def get_mass_flow(self, thrust: float) -> float:
        r"""
        Calculate mass flow rate.

        Equation:
        $\dot{m} = \frac{T}{I_{sp} g_0}$

        Parameters
        ----------
        thrust : float
            Actual thrust produced (N).

        Returns
        -------
        float
            Mass flow rate (kg/s).
        """
        if self.isp and self.isp > 0:
            g0: float = 9.80665
            return thrust / (self.isp * g0)
        return 0.0


class ChemicalThruster(Thruster):
    """
    Chemical Thruster model.

    Models On/Off behavior or PWM-averaged thrust with minimum on-time constraints.

    Parameters
    ----------
    max_thrust : float, optional
        Maximum thrust (N). Default is 10.0.
    isp : float, optional
        Specific impulse (s). Default is 300.0.
    min_on_time : float, optional
        Minimum valve open time (s). Default is 0.010.
    name : str, optional
        Actuator name. Default is "ChemThruster".
    """

    def __init__(
        self,
        max_thrust: float = 10.0,
        isp: float = 300.0,
        min_on_time: float = 0.010,
        name: str = "ChemThruster",
    ) -> None:
        """
        Initialize chemical thruster.

        Parameters
        ----------
        max_thrust : float, optional
            Maximum thrust (N). Default 10.0.
        isp : float, optional
            Specific impulse (s). Default 300.0.
        min_on_time : float, optional
            Minimum valve open time (s). Default 0.010.
        name : str, optional
            Actuator name. Default "ChemThruster".
        """
        self.min_on_time: float = min_on_time
        mib: float = max_thrust * min_on_time
        super().__init__(max_thrust=max_thrust, isp=isp, min_impulse_bit=mib, name=name)

    def command(
        self, thrust_cmd: float | None = None, dt: float | None = None, *args: Any, **kwargs: Any
    ) -> float:
        """
        Considers PWM constraints for chemical valve.

        If the commanded thrust implies an on-time < min_on_time, it is zeroed.

        Parameters
        ----------
        thrust_cmd : float
            Commanded thrust (N).
        dt : float, optional
            Control period (s).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            Delivered thrust (N).
        """
        if thrust_cmd is None and args:
            thrust_cmd = float(args[0])
        thrust: float = super().command(thrust_cmd, dt=dt, **kwargs)

        if dt is not None and self.min_on_time > 0 and abs(thrust) > 1e-9:
            required_on_time: float = (abs(thrust) / self.max_thrust) * dt

            if required_on_time < self.min_on_time:
                thrust = 0.0

        return thrust


class ElectricThruster(Thruster):
    """
    Electric Thruster model (e.g., Hall Effect, Ion).

    Power-limited thrust generation.

    Parameters
    ----------
    max_thrust : float, optional
        Maximum thrust (N). Default is 0.1.
    isp : float, optional
        Specific impulse (s). Default is 1500.0.
    power_efficiency : float, optional
        Electrical to Jet power efficiency (eta). Default is 0.6.
    name : str, optional
        Actuator name. Default is "ElecThruster".
    """

    def __init__(
        self,
        max_thrust: float = 0.1,
        isp: float = 1500.0,
        power_efficiency: float = 0.6,
        name: str = "ElecThruster",
    ) -> None:
        """
        Initialize electric thruster.

        Parameters
        ----------
        max_thrust : float, optional
            Maximum thrust (N). Default 0.1.
        isp : float, optional
            Specific impulse (s). Default 1500.0.
        power_efficiency : float, optional
            Electrical to Jet power efficiency (eta). Default 0.6.
        name : str, optional
            Actuator name. Default "ElecThruster".
        """
        super().__init__(max_thrust=max_thrust, isp=isp, name=name)
        self.power_efficiency: float = power_efficiency
        self.g0: float = 9.80665

    def get_power_consumption(self, thrust: float) -> float:
        r"""
        Calculate electrical power requirement.

        Equation:
        $P_{in} = \frac{T I_{sp} g_0}{2 \eta}$

        Parameters
        ----------
        thrust : float
            Produced thrust (N).

        Returns
        -------
        float
            Electrical power consumption (W).
        """
        if self.power_efficiency <= 0:
            return float("inf")

        if self.isp is None:
            return 0.0
        ve = self.isp * self.g0
        return float(thrust * ve / (2 * self.power_efficiency))


class ThrusterCluster:
    """
    A collection of thrusters with defined allocation logic.

    Maps generalized 6-DOF force/torque commands to individual thruster outputs.

    Parameters
    ----------
    thrusters : list[Thruster]
        List of thruster objects.
    positions : np.ndarray
        (N, 3) positions of thrusters in body frame (m).
    directions : np.ndarray
        (N, 3) thrust unit vectors in body frame.
    """

    def __init__(
        self, thrusters: list[Thruster], positions: np.ndarray, directions: np.ndarray
    ) -> None:
        self.thrusters = thrusters
        self.N = len(thrusters)
        self.pos = np.array(positions)
        self.dir = np.array(directions)

        # Force_i = T_i * dir_i
        # Torque_i = pos_i x (T_i * dir_i)
        self.A = np.zeros((6, self.N))
        for i in range(self.N):
            self.A[0:3, i] = self.dir[i]
            self.A[3:6, i] = np.cross(self.pos[i], self.dir[i])

        # Default allocator
        from opengnc.actuators.allocation import PseudoInverseAllocator

        self.allocator = PseudoInverseAllocator(self.A)

    def command(
        self, force_torque_cmd: np.ndarray | None = None, dt: float | None = None, *args: Any
    ) -> np.ndarray:
        """
        Distribute 6-DOF force/torque command to individual thrusters.

        Parameters
        ----------
        force_torque_cmd : np.ndarray
            Desired [Fx, Fy, Fz, Tx, Ty, Tz] in body frame (N, Nm).
        dt : float, optional
            Time step for MIB and duty cycle checks (s).

        Returns
        -------
        np.ndarray
            Delivered thrusts for each thruster in the cluster (N).
        """
        if force_torque_cmd is None:
            if not args:
                raise ValueError("force_torque_cmd is required.")
            force_torque_cmd = np.asarray(args[0])
        thrust_cmds = self.allocator.allocate(force_torque_cmd)

        # Apply individual thruster constraints
        delivered_thrusts = []
        for i, cmd in enumerate(thrust_cmds):
            cmd_clamped = max(0.0, cmd)
            delivered = self.thrusters[i].command(cmd_clamped, dt=dt)
            delivered_thrusts.append(delivered)

        return np.array(delivered_thrusts)
