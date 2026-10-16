import csv
import json
from typing import Any


class SimulationLogger:
    """
    Simulation replay and logging interface.
    Records state history and outputs to common formats like JSON, CSV.
    (HDF5 support optionally using h5py if installed).
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.history: list[dict[str, Any]] = []

    def log(
        self,
        t: float,
        state: Any,
        measurements: Any = None,
        estimates: Any = None,
        commands: Any = None,
    ) -> None:
        """
        Record a simulation step.

        Parameters
        ----------
        t : float
            Simulation time.
        state : Any
            True state of the system (e.g. state vector).
        measurements : Any, optional
            Sensor measurements at this time step.
        estimates : Any, optional
            Filter estimates at this time step.
        commands : Any, optional
            Control commands sent to actuators.
        """
        entry = {
            "time": t,
            "state": state,
        }
        if measurements is not None:
            entry["measurements"] = measurements
        if estimates is not None:
            entry["estimates"] = estimates
        if commands is not None:
            entry["commands"] = commands

        self.history.append(entry)

    def save_json(self) -> None:
        """Saves the logged history to a JSON file."""
        with open(self.filename + ".json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

    def save_csv(self) -> None:
        """Saves the log's top-level keys to a CSV file."""
        if not self.history:
            return
        keys: set[str] = set()
        for h in self.history:
            keys.update(h.keys())

        with open(self.filename + ".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def save_hdf5(self) -> None:
        """Saves the logged history to an HDF5 file (requires h5py)."""
        try:
            import h5py
            import numpy as np

            with h5py.File(self.filename + ".h5", "w") as f:
                # Basic implementation supporting 1D numpy states
                times = [h["time"] for h in self.history]
                f.create_dataset("time", data=np.array(times))

                # Try to stack states if they are arrays or list of numbers
                try:
                    states = np.array([h["state"] for h in self.history])
                    f.create_dataset("state", data=states)
                except Exception:
                    pass
        except ImportError:
            print("Warning: h5py not installed. Cannot save to HDF5 format.")
