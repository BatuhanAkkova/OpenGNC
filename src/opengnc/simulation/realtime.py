import time
from typing import Any

from .simulator import MissionSimulator


class RealTimeSimulator(MissionSimulator):
    """
    Real-time simulation synchronizing simulation time to wall-clock time.
    Supports Hardware-In-the-Loop (HIL) or Software-In-the-Loop (SIL) testing.
    """

    def __init__(self, *args: Any, rtf: float = 1.0, **kwargs: Any) -> None:
        """
        Initialize real-time simulator.

        Parameters
        ----------
        rtf : float
            Real-Time Factor.
            rtf = 1.0 means simulation runs exactly at wall-clock speed.
            rtf = 2.0 means simulation runs twice as fast.
        *args : Any
            Passed to MissionSimulator.
        **kwargs : Any
            Passed to MissionSimulator.
        """
        super().__init__(*args, **kwargs)
        self.rtf = rtf
        self.wall_clock_start = 0.0
        self.sim_clock_start = 0.0

    def initialize(self, t0: float, initial_state: Any) -> None:
        super().initialize(t0, initial_state)
        self.sim_clock_start = t0
        self.wall_clock_start = time.time()

    def run(self, t_end: float, dt: float) -> None:
        """
        Runs the simulation loop until the end time, matching wall-clock pace.

        Parameters
        ----------
        t_end : float
            Simulation end time.
        dt : float
            Simulation time step.
        """
        while self.time <= t_end:
            # step logic
            super().step(dt)

            # calculate required delay
            elapsed_sim = self.time - self.sim_clock_start
            expected_wall_clock = elapsed_sim / self.rtf

            elapsed_wall = time.time() - self.wall_clock_start
            sleep_time = expected_wall_clock - elapsed_wall

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -dt:
                # Issue a warning if we consistenty fall behind schedule
                print(f"Warning: Real-time deadline missed by {-sleep_time:.3f}s")
