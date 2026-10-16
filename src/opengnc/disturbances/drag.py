"""
Lumped atmospheric drag model for spacecraft acceleration.
"""

from typing import Any

import numpy as np


class LumpedDrag:
    r"""
    Atmospheric Drag Model.

    Acceleration Equation:
    $\mathbf{a}_d = -\frac{1}{2} \rho v_{rel}^2 \frac{C_d A}{m} \hat{\mathbf{v}}_{rel}$

    Parameters
    ----------
    density_model : Any
        Atmospheric density provider.
    co_rotate : bool, optional
        Atmospheric co-rotation with planet. Default True.
    """

    def __init__(self, density_model: Any, co_rotate: bool = True) -> None:
        """Initialize drag model with density source."""
        self.density_model = density_model
        self.co_rotate = co_rotate

    def get_acceleration(
        self, r_eci: np.ndarray, v_eci: np.ndarray, jd: float, mass: float, area: float, cd: float
    ) -> np.ndarray:
        r"""
        Calculate instantaneous drag acceleration vector.

        Formula:
        $\mathbf{a}_d = -\frac{1}{2} \rho v_{rel}^2 \frac{C_d A}{m} \hat{\mathbf{v}}_{rel}$

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position vector (m).
        v_eci : np.ndarray
            ECI Velocity vector (m/s).
        jd : float
            Julian Date.
        mass : float
            Total spacecraft mass (kg).
        area : float
            Effective cross-sectional area ($m^2$).
        cd : float
            Drag coefficient.

        Returns
        -------
        np.ndarray
            Acceleration vector ($m/s^2$).
        """
        r_val = np.asarray(r_eci)
        v_val = np.asarray(v_eci)

        rho = self.density_model.get_density(r_val, jd)

        if self.co_rotate:
            # Earth rotation vector approx (rad/s)
            w_earth = np.array([0, 0, 7.2921159e-5])
            v_rel = v_val - np.cross(w_earth, r_val)
        else:
            v_rel = v_val

        v_rel_norm = np.linalg.norm(v_rel)
        if v_rel_norm < 1e-6:
            return np.zeros(3, dtype=float)

        # Acceleration calculation
        acc_mag = 0.5 * rho * v_rel_norm**2 * cd * (area / mass)
        return np.asarray(-acc_mag * (v_rel / v_rel_norm))
