"""
Space weather index management (F10.7, Ap, Kp) for disturbance models.
"""

from typing import Any

import numpy as np


class SpaceWeather:
    """
    Solar and Geomagnetic Activity Indices Manager.

    Coordinates F10.7 (solar flux) and Ap/Kp (geomagnetic) indices used by
    thermospheric and radiation models.

    Parameters
    ----------
    f107 : float, optional
        Daily solar flux index at 10.7cm (sfu). Default 150.0.
    f107_avg : float, optional
        81-day centered mean solar flux (sfu). Default 150.0.
    ap : float, optional
        Planetary equivalent amplitude index (geomagnetic). Default 15.0.
    """

    def __init__(self, f107: float = 150.0, f107_avg: float = 150.0, ap: float = 15.0) -> None:
        """Initialize indices and compute derived Kp."""
        self.f107 = f107
        self.f107_avg = f107_avg
        self.ap = ap
        self.kp = self._ap_to_kp(ap)

    def _ap_to_kp(self, ap: float) -> float:
        """
        Convert planetary Ap to Kp index.

        Parameters
        ----------
        ap : float
            Ap index.

        Returns
        -------
        float
            Estimated Kp index [0, 9].
        """
        if ap <= 0:
            return 0.0
        return float(3.0 * np.log2(ap / 2.0 + 1.0) / 2.0)

    def get_indices(self, date: Any | None = None) -> dict[str, float]:
        """
        Retrieve indices for a given epoch.

        Parameters
        ----------
        date : Optional[Any], optional
            Target epoch. Currently returns static values.

        Returns
        -------
        Dict[str, float]
            Indices dictionary.
        """
        return {"f107": self.f107, "f107_avg": self.f107_avg, "ap": self.ap, "kp": self.kp}

    def set_solar_flux(self, f107: float, f107_avg: float | None = None) -> None:
        """
        Update local solar flux parameters.

        Parameters
        ----------
        f107 : float
            Daily solar flux (sfu).
        f107_avg : Optional[float], optional
            Average solar flux. Defaults to `f107` if None.
        """
        self.f107 = float(f107)
        self.f107_avg = float(f107_avg) if f107_avg is not None else self.f107
