"""
Reaction wheel momentum desaturation using magnetic torque.
"""

import numpy as np


class CrossProductLaw:
    r"""
    Reaction wheel momentum desaturation using the cross-product law.

    This controller calculates a magnetic dipole moment ``m`` such that the
    resulting magnetic torque ``T = m x B`` opposes the component of the
    angular momentum error perpendicular to the magnetic field.

    Control law: ``m = k * (H_err x B) / ||B||^2``.
    """

    def __init__(self, gain: float, max_dipole: float | None = None) -> None:
        """Initialize the momentum dumping controller."""
        self.gain = gain
        self.max_dipole = max_dipole

    def calculate_control(
        self, h_error: np.ndarray | list[float], b_field: np.ndarray | list[float]
    ) -> np.ndarray:
        """Calculate the required magnetic dipole moment."""
        h_vec = np.asarray(h_error, dtype=float)
        b_vec = np.asarray(b_field, dtype=float)
        b_sq = np.dot(b_vec, b_vec)

        if b_sq < 1e-18:
            return np.zeros(3, dtype=float)

        dipole_moment = (self.gain / b_sq) * np.cross(h_vec, b_vec)
        if self.max_dipole is not None:
            norm_m = np.linalg.norm(dipole_moment)
            if norm_m > self.max_dipole:
                dipole_moment *= self.max_dipole / norm_m

        return np.asarray(dipole_moment, dtype=float)
