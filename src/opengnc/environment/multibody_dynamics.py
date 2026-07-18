"""
Circular Restricted Three-Body Problem (CR3BP) dynamics.
"""

import numpy as np


class CR3BP:
    r"""
    Circular Restricted Three-Body Problem Dynamics.

    Models the motion of a negligible mass under the influence of two
    massive bodies (primaries) in a circular orbit about their barycenter.

    Parameters
    ----------
    mu : float
        Mass parameter $\mu = m_2 / (m_1 + m_2)$.
    """

    def __init__(self, mu: float) -> None:
        """Initialize with mass parameter."""
        self.mu = mu

    def get_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""
        Calculate state derivatives in the rotating frame.

        Parameters
        ----------
        t : float
            Time (normalized).
        state : np.ndarray
            Current state $[x, y, z, vx, vy, vz]$.

        Returns
        -------
        np.ndarray
            Derivative vector $[\dot{x}, \dot{y}, \dot{z}, \dot{vx}, \dot{vy}, \dot{vz}]$.
        """
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)

        # Potential derivatives
        ax = x - (1 - self.mu) * (x + self.mu) / r1**3 - self.mu * (x - 1 + self.mu) / r2**3
        ay = y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3
        az = -(1 - self.mu) * z / r1**3 - self.mu * z / r2**3

        # Add Coriolis acceleration
        acc_x = ax + 2 * vy
        acc_y = ay - 2 * vx
        acc_z = az

        return np.array([vx, vy, vz, acc_x, acc_y, acc_z])

    def calculate_jacobi_constant(self, state: np.ndarray) -> float:
        r"""
        Calculate the Jacobi constant (Integral of motion).

        $C = (x^2 + y^2) + 2\frac{1-\mu}{r_1} + 2\frac{\mu}{r_2} - (vx^2 + vy^2 + vz^2)$

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        float
            Jacobi constant value.
        """
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)

        v_sq = vx**2 + vy**2 + vz**2
        potential = (x**2 + y**2) + 2 * (1 - self.mu) / r1 + 2 * self.mu / r2

        return float(potential - v_sq)
