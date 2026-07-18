"""
Base classes for external tool interfaces in OpenGNC.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ExternalTool(ABC):
    """
    Abstract base class for external tool integrations.
    """

    @abstractmethod
    def connect(self, **kwargs: Any) -> bool:
        """Establish connection or initialize the tool."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up resources and disconnect."""
        pass


class ExternalPropagator(ExternalTool):
    """
    Abstract base class for external propagators (GMAT, Orekit, etc.).
    """

    @abstractmethod
    def propagate(
        self,
        initial_state: np.ndarray,
        start_jd: float,
        duration_sec: float,
        step_sec: float,
    ) -> dict[str, np.ndarray]:
        """
        Run propagation in the external tool.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state [x, y, z, vx, vy, vz] in meters and m/s.
        start_jd : float
            Start time in Julian Date.
        duration_sec : float
            Propagation duration in seconds.
        step_sec : float
            Output time step in seconds.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'times' (Julian Dates) and 'states' (nx6 array).
        """
        pass
