from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any


class Event:
    """
    A single discrete event to be processed at a specific time.
    """

    def __init__(self, t: float, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """
        Initialize a discrete event.

        Parameters
        ----------
        t : float
            Simulation time at which the event should trigger.
        callback : Callable
            Function to be executed when the event triggers.
        """
        self.t = t
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __lt__(self, other: Event) -> bool:
        return self.t < other.t

    def execute(self) -> Any:
        """Executes the event callback."""
        return self.callback(*self.args, **self.kwargs)


class EventQueue:
    """
    Priority queue managing discrete events in the simulation.
    Handles maneuver scheduling and mode transitions.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    def schedule(self, t: float, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """
        Schedule a new event.

        Parameters
        ----------
        t : float
            Time at which to run the callback.
        callback : Callable
            The function to execute.
        """
        heapq.heappush(self._events, Event(t, callback, *args, **kwargs))

    def has_events(self) -> bool:
        """Returns True if there are pending events."""
        return len(self._events) > 0

    def next_event_time(self) -> float:
        """Returns the time of the next pending event, or infinity if empty."""
        if not self._events:
            return float("inf")
        return self._events[0].t

    def process_until(self, current_time: float) -> None:
        """
        Process all events scheduled up to the provided current_time.

        Parameters
        ----------
        current_time : float
            The current simulation time.
        """
        while self._events and self._events[0].t <= current_time:
            event = heapq.heappop(self._events)
            event.execute()
