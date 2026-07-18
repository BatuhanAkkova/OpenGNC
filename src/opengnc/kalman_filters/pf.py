"""
Particle Filter (Sequential Importance Resampling) for non-Gaussian, nonlinear systems.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np


class ParticleFilter:
    """
    Bootstrap particle filter.

    Represents the posterior distribution using weighted particles.
    """

    def __init__(self, dim_x: int, dim_z: int, num_particles: int = 1000) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, dim_x))
        self.weights = np.ones(num_particles) / num_particles
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def initialize_particles(self, x_mean: np.ndarray, p_cov: np.ndarray) -> None:
        """Initialize particles from a multivariate Gaussian distribution."""
        self.particles = cast(
            np.ndarray,
            np.random.multivariate_normal(x_mean, p_cov, self.num_particles),
        )
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(
        self,
        dt: float,
        fx_func: Callable,
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Propagate each particle and inject process noise."""
        q_curr = q_mat if q_mat is not None else self.Q
        for i in range(self.num_particles):
            self.particles[i] = cast(np.ndarray, fx_func(self.particles[i], dt, **kwargs))
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), q_curr)
            self.particles[i] += noise

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Reweight particles from the latest measurement and resample if needed."""
        r_curr = r_mat if r_mat is not None else self.R
        inv_r = np.linalg.inv(r_curr)
        det_r = np.linalg.det(r_curr)
        norm_factor = 1.0 / np.sqrt((2.0 * np.pi) ** self.dim_z * det_r)

        for i in range(self.num_particles):
            z_pred = cast(np.ndarray, hx_func(self.particles[i], **kwargs))
            diff = z - z_pred
            prob = norm_factor * np.exp(-0.5 * (diff.T @ inv_r @ diff))
            self.weights[i] *= prob

        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        if self.neff() < self.num_particles / 2:
            self.resample()

    def resample(self) -> None:
        """Resample particles using systematic resampling."""
        cum_sum = np.cumsum(self.weights)
        cum_sum[-1] = 1.0
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        indices = np.zeros(self.num_particles, dtype=int)

        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cum_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def neff(self) -> float:
        """Calculate the effective number of particles."""
        return float(1.0 / np.sum(np.square(self.weights)))

    @property
    def x(self) -> np.ndarray:
        """Return the weighted mean state."""
        return cast(np.ndarray, np.average(self.particles, weights=self.weights, axis=0))

    @property
    def P(self) -> np.ndarray:
        """Return the weighted error covariance matrix."""
        x_mean = self.x
        diff = self.particles - x_mean
        return cast(np.ndarray, (self.weights * diff.T) @ diff)
