"""
Navigation using GNSS (GPS) position and velocity measurements.
"""


from typing import Any

import numpy as np

from .orbit_determination import OrbitDeterminationEKF


class GPSNavigation(OrbitDeterminationEKF):
    r"""
    GNSS-based Navigation using absolute Position and Velocity measurements.

    Processes PVT (Position, Velocity, Time) solutions from a receiver
    to correct the integrated orbital state. Inherits dynamics from
    `OrbitDeterminationEKF`.
    """

    def update_gps(
        self,
        r_meas: np.ndarray,
        v_meas: np.ndarray | None = None,
        gps_cov: np.ndarray | None = None,
        **kwargs: Any
    ) -> None:
        r"""
        Update state estimate using GNSS Cartesian measurements.

        Parameters
        ----------
        r_meas : np.ndarray
            Measured ECI position (m).
        v_meas : np.ndarray, optional
            Measured ECI velocity (m/s).
        gps_cov : np.ndarray, optional
            Direct measurement covariance matrix.
        **kwargs : dict
            Additional parameters (e.g., R_gps).
        """
        rv = np.asarray(r_meas)

        # Handle custom R passed via kwargs or gps_cov
        r_custom = gps_cov if gps_cov is not None else kwargs.get("R_gps")

        if v_meas is not None:
            vv = np.asarray(v_meas)
            z_vec = np.concatenate([rv, vv])

            def hx(x: np.ndarray) -> np.ndarray:
                return x  # Identity mapping

            def h_jac(x: np.ndarray) -> np.ndarray:
                return np.eye(6)

            if r_custom is not None:
                r_mat = np.asarray(r_custom)
            else:
                # Scaled default covariance
                r_mat = np.eye(6)
                # Ensure self.ekf.R is captured correctly even if not initialized as expected
                r_pos = self.ekf.R if (hasattr(self.ekf, 'R') and self.ekf.R.size > 0) else np.eye(3)
                r_mat[:3, :3] = r_pos
                r_mat[3:, 3:] = r_pos * 0.1  # Velocity noise heuristic

            self.ekf.update(z_vec, hx, h_jac, r_mat=r_mat)
        else:
            # Position only update
            self.update(rv)




