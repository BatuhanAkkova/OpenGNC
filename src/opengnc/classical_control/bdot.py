"""
B-Dot controller for spacecraft magnetic detumbling.
"""

import numpy as np


class BDot:
    r"""
    B-Dot controller for magnetic detumbling.

    Control Law:
    $\mathbf{m} = -K_{gain} \dot{\mathbf{B}}$

    Parameters
    ----------
    gain : float
        Feedback gain $K$ ($Am^2 s / T$).
    """

    def __init__(self, gain: float) -> None:
        """Initialize the B-Dot controller."""
        self.gain = gain

    def calculate_control(self, b_dot: np.ndarray | list[float]) -> np.ndarray:
        r"""
        Calculate the required magnetic dipole moment from the B-field rate.

        Parameters
        ----------
        b_dot : np.ndarray or list
            The time derivative of the magnetic field vector ($\dot{B}$) in
            the Body frame (3,). Units: [T/s].

        Returns
        -------
        np.ndarray
            Magnetic dipole moment vector $m$ in Body frame (3,).
            Units: $[A \cdot m^2]$.
        """
        # Standard law: m = -K * B_dot
        return -self.gain * np.asarray(b_dot, dtype=float)

    def calculate_control_discrete(
        self,
        b_field_curr: np.ndarray | list[float],
        b_field_prev: np.ndarray | list[float],
        dt: float,
    ) -> np.ndarray:
        r"""
        Calculate B-Dot control using discrete finite differences.

        Useful when only B-field measurements are available instead of explicit rates.

        Parameters
        ----------
        b_field_curr : np.ndarray or list
            Current magnetic field vector measurement (Body frame). Units: [T].
        b_field_prev : np.ndarray or list
            Previous magnetic field vector measurement (Body frame). Units: [T].
        dt : float
            Time step between measurements (s).

        Returns
        -------
        np.ndarray
            Magnetic dipole moment vector $m$ in Body frame (3,).
        """
        if dt <= 0:
            return np.zeros(3)

        # Estimate B_dot via backward difference
        b_dot_est = (np.asarray(b_field_curr) - np.asarray(b_field_prev)) / dt
        return self.calculate_control(b_dot_est)
