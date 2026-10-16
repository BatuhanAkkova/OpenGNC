from collections.abc import Callable
from typing import Any

from .events import EventQueue
from .logging import SimulationLogger

# Type aliases for simulation callbacks
PropagatorFunc = Callable[[float, Any, float, Any], Any]
SensorFunc = Callable[[float, Any], Any]
EstimatorFunc = Callable[[float, Any], Any]
ControllerFunc = Callable[[float, Any], Any]


class MissionSimulator:
    r"""
    High-Fidelity Mission Simulator.

    Loop Sequence:
    1. Update Sense: $\mathbf{y} = h(t, \mathbf{x}) + \mathbf{v}$
    2. Estimate: $\hat{\mathbf{x}} = f(t, \mathbf{y})$
    3. Control: $\mathbf{u} = g(t, \hat{\mathbf{x}})$
    4. Propagate: $\dot{\mathbf{x}} = f(t, \mathbf{x}, \mathbf{u})$

    Parameters
    ----------
    propagator : PropagatorFunc
        Truth propagation: $(t, x, dt, u) \to x_{new}$
    sensor_model : Optional[SensorFunc]
        Measurement model: $(t, x) \to y$
    estimator : Optional[EstimatorFunc]
        State estimation: $(t, y) \to \hat{x}$
    controller : Optional[ControllerFunc]
        Control law: $(t, \hat{x}) \to u$
    logger : Optional[SimulationLogger]
        Telemetry recording.
    """

    def __init__(
        self,
        propagator: PropagatorFunc,
        sensor_model: SensorFunc | None = None,
        estimator: EstimatorFunc | None = None,
        controller: ControllerFunc | None = None,
        logger: SimulationLogger | None = None,
    ) -> None:
        """Initialize simulator core and event management."""
        self.propagator = propagator
        self.sensor_model = sensor_model
        self.estimator = estimator
        self.controller = controller

        self.logger = logger
        self.event_queue = EventQueue()

        self.time: float = 0.0
        self.state: Any = None

    def initialize(self, t0: float, initial_state: Any) -> None:
        """
        Set the simulation starting epoch and state.

        Parameters
        ----------
        t0 : float
            Start time (s).
        initial_state : Any
            Initial truth state vector or object.
        """
        self.time = t0
        self.state = initial_state

    def schedule_event(
        self, t: float, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Register a discrete event for future execution.

        Parameters
        ----------
        t : float
            Target execution time (s).
        callback : Callable
            Function to execute at time $t$.
        """
        self.event_queue.schedule(t, callback, *args, **kwargs)

    def step(self, dt: float) -> None:
        """
        Execute a single fixed-step simulation frame.

        Sequence:
        1. Process pending discrete events.
        2. Generate synthetic measurements (Sense).
        3. Solve for state estimate (Estimate).
        4. Compute control effort (Control).
        5. Log data series.
        6. Advance physics (Propagate).

        Parameters
        ----------
        dt : float
            Simulation time step (s).
        """
        # 1. Process discrete events
        self.event_queue.process_until(self.time)

        # 2. Sense
        meas = self.sensor_model(self.time, self.state) if self.sensor_model else None

        # 3. Estimate
        est = self.estimator(self.time, meas) if self.estimator else None

        # 4. Control
        # Fallback to perfect knowledge if no estimator
        ctrl_ctx = est if est is not None else self.state
        u = self.controller(self.time, ctrl_ctx) if self.controller else None

        # 5. Record
        if self.logger:
            self.logger.log(self.time, self.state, meas, est, u)

        # 6. Propagate Truth
        self.state = self.propagator(self.time, self.state, dt, u)

        self.time += dt

    def run(self, t_end: float, dt: float) -> Any:
        """
        Execute the simulation loop until the termination time.

        Parameters
        ----------
        t_end : float
            Simulation stop time (s).
        dt : float
            Fixed integration step (s).

        Returns
        -------
        Any
            Returns logger history if available, otherwise simulation end state.
        """
        while self.time <= t_end:
            self.step(dt)

        return self.logger.history if self.logger else self.state
