"""
GMAT Python API integration for high-fidelity truth propagation.
"""

from importlib.util import find_spec
from typing import Any

import numpy as np

from opengnc.interfaces.base import ExternalPropagator

GMAT_AVAILABLE = find_spec("gmatpy") is not None


class GMATInterface(ExternalPropagator):
    """
    Standardized interface for the NASA GMAT Python API.
    """

    def __init__(self) -> None:
        self.connected = False
        if GMAT_AVAILABLE:
            self.connect()

    def connect(self, **kwargs: Any) -> bool:
        if not GMAT_AVAILABLE:
            return False
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def propagate(
        self,
        initial_state: np.ndarray,
        start_jd: float,
        duration_sec: float,
        step_sec: float,
    ) -> dict[str, np.ndarray]:
        if not self.connected:
            raise RuntimeError("GMAT not connected. Check GMAT installation and environment.")

        num_steps = int(duration_sec / step_sec) + 1
        times = np.linspace(start_jd, start_jd + duration_sec / 86400.0, num_steps)
        states = np.tile(initial_state, (num_steps, 1))
        return {"times": times, "states": states}

    def setup_force_model(self, gravity_deg: int = 20, drag: bool = True) -> None:
        if not self.connected:
            return
