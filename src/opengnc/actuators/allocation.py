"""
Control allocation algorithms mapping generalized forces to actuator commands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np


class ControlAllocator(ABC):
    """
    Abstract base class for control allocation logic.

    Maps desired generalized forces (force/torque) to individual actuator commands.

    Parameters
    ----------
    actuator_matrix : np.ndarray
        The (m x n) matrix A where Y = A * u.
        m = degrees of freedom (e.g., 3 for torque).
        n = number of actuators.
    """

    def __init__(self, actuator_matrix: np.ndarray) -> None:
        self.A = np.array(actuator_matrix)
        self.m, self.n = self.A.shape

    @abstractmethod
    def allocate(self, force_torque_cmd: np.ndarray) -> np.ndarray:
        """
        Allocate generalized force/torque to individual actuators.

        Parameters
        ----------
        force_torque_cmd : np.ndarray
            Desired 6-DOF force and torque vector.

        Returns
        -------
        np.ndarray
            Individual actuator commands.
        """
        pass


class PseudoInverseAllocator(ControlAllocator):
    """
    Control allocation using the Moore-Penrose pseudo-inverse.

    Suitable for over-determined systems (more actuators than DOFs).
    Minimizes the L2-norm of the actuator commands: u = A^# * f.

    Parameters
    ----------
    effectiveness_matrix : np.ndarray
        The (m x n) matrix mapping actuator outputs to forces/torques.
    """

    def __init__(self, effectiveness_matrix: np.ndarray) -> None:
        self.A = effectiveness_matrix
        # Precompute pseudo-inverse
        self.A_pinv = np.linalg.pinv(self.A)

    def allocate(self, force_torque_cmd: np.ndarray) -> np.ndarray:
        """
        Perform allocation using pseudo-inverse.

        Parameters
        ----------
        force_torque_cmd : np.ndarray
            Desired force/torque vector.

        Returns
        -------
        np.ndarray
            Actuator commands vector.
        """
        return cast(np.ndarray, self.A_pinv @ force_torque_cmd)


class SingularRobustAllocator(ControlAllocator):
    """
    Singular Robust Inverse (SR-Inverse) allocator for CMGs.

    Adds a regularization term to avoid extremely high actuator rates near singularities.
    u = A^T * (A * A^T + lambda * I)^-1 * Y.

    Parameters
    ----------
    actuator_matrix : np.ndarray
        The control effectiveness matrix.
    epsilon : float, optional
        Threshold for manipulability measure. Default is 0.01.
    lambda0 : float, optional
        Maximum regularization weight. Default is 0.01.
    """

    def __init__(self, actuator_matrix: np.ndarray, epsilon: float = 0.01, lambda0: float = 0.01) -> None:
        super().__init__(actuator_matrix)
        self.epsilon = epsilon
        self.lambda0 = lambda0

    def allocate(self, desired_output: np.ndarray, A_current: np.ndarray | None = None) -> np.ndarray:
        """
        Perform SR-Inverse allocation.

        Parameters
        ----------
        desired_output : np.ndarray
            Desired torque or force.
        A_current : np.ndarray, optional
            Current Jacobian matrix if time-varying. Defaults to self.A.

        Returns
        -------
        np.ndarray
            Actuator rate commands.
        """
        A = A_current if A_current is not None else self.A

        # Calculate manipulability measure (determinant of AA^T)
        m_measure = float(np.linalg.det(A @ A.T))

        # Regularization parameter lambda
        if m_measure < self.epsilon:
            lam = self.lambda0 * (1 - m_measure / self.epsilon) ** 2
        else:
            lam = 0.0

        # SR-Inverse: A^T (A A^T + lam * I)^-1
        m = A.shape[0]
        inv_term = np.linalg.inv(A @ A.T + lam * np.eye(m))
        u = A.T @ inv_term @ desired_output
        return cast(np.ndarray, u)


class NullMotionManager:
    """
    Manages null-motion to redistribute actuator states without affecting net output.

    u_net = u_alloc + u_null
    u_null = (I - A+ * A) * z

    Parameters
    ----------
    actuator_matrix : np.ndarray
        The control effectiveness matrix.
    """

    def __init__(self, actuator_matrix: np.ndarray) -> None:
        self.A = np.array(actuator_matrix)
        self.m, self.n = self.A.shape
        self.I = np.eye(self.n)

    def get_null_projection(self, A_current: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the null-space projection matrix (I - A+ * A).

        Parameters
        ----------
        A_current : np.ndarray, optional
            Current Jacobian/effectiveness matrix.

        Returns
        -------
        np.ndarray
            Null-space projection matrix.
        """
        A = A_current if A_current is not None else self.A
        A_pinv = np.linalg.pinv(A)
        return cast(np.ndarray, self.I - A_pinv @ A)

    def apply_null_command(self, u_base: np.ndarray, z: np.ndarray, A_current: np.ndarray | None = None) -> np.ndarray:
        """
        Add null-motion component to base command.

        Parameters
        ----------
        u_base : np.ndarray
            Base actuator commands (e.g., from pseudo-inverse).
        z : np.ndarray
            Desired secondary goal (e.g., move gimbals away from limits).
        A_current : np.ndarray, optional
            Current Jacobian.

        Returns
        -------
        np.ndarray
            Combined actuator command.
        """
        P = self.get_null_projection(A_current)
        return cast(np.ndarray, u_base + P @ z)




