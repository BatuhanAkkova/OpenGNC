from typing import Any

from .simulator import (
    ControllerFunc,
    EstimatorFunc,
    MissionSimulator,
    PropagatorFunc,
    SensorFunc,
    SimulationLogger,
)


class ConstellationSimulator:
    """
    Multi-Agent / Constellation Simulator.

    Manages simultaneous simulation of multiple spacecraft, enabling
    coordinated formation flying or large network analysis.

    Parameters
    ----------
    num_satellites : int
        Total spacecraft count.
    propagator : PropagatorFunc
        Joint or vectorized propagator function.
    sensor_model : Optional[SensorFunc]
        Measurement model.
    estimator : Optional[EstimatorFunc]
        Centralized or multi-state estimator.
    controller : Optional[ControllerFunc]
        Coordinating control logic.
    logger : Optional[SimulationLogger]
        Data recording utility.
    """

    def __init__(
        self,
        num_satellites: int,
        propagator: PropagatorFunc,
        sensor_model: SensorFunc | None = None,
        estimator: EstimatorFunc | None = None,
        controller: ControllerFunc | None = None,
        logger: SimulationLogger | None = None,
    ) -> None:
        """Initialize simulation for spacecraft N members."""
        self.num_satellites = num_satellites
        self.simulator = MissionSimulator(
            propagator=propagator,
            sensor_model=sensor_model,
            estimator=estimator,
            controller=controller,
            logger=logger,
        )

    def initialize(self, t0: float, initial_states: list[Any]) -> None:
        """
        Initialize starting conditions for all members.

        Parameters
        ----------
        t0 : float
            Start simulation time (s).
        initial_states : List[Any]
            Container of starting states for each spacecraft.
        """
        if len(initial_states) != self.num_satellites:
            raise ValueError(
                f"Expected {self.num_satellites} initial states, got {len(initial_states)}"
            )

        self.simulator.initialize(t0, initial_states)

    def run(self, t_end: float, dt: float) -> Any:
        """
        Execute the constrained multi-body simulation.

        Parameters
        ----------
        t_end : float
            Termination time (s).
        dt : float
            Physics update rate (s).

        Returns
        -------
        Any
            Aggregate simulation results.
        """
        return self.simulator.run(t_end, dt)




