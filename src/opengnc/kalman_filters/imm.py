"""
Interacting Multiple Model (IMM) Filter for switching-mode systems.
"""

import inspect
from typing import Any

import numpy as np


class IMM:
    """
    Interacting Multiple Model (IMM) Filter.

    Estimates the state of a system that can switch between multiple discrete
    modes (models). Ideal for tracking maneuvering targets where dynamics
    switch between models like constant velocity and constant acceleration.

    Parameters
    ----------
    filters : list
        List of filter objects (e.g., KF, EKF, UKF).
    transition_matrix : np.ndarray
        Model transition probability matrix (N x N), where $T_{ij} = P(M_j | M_i)$.
    """

    def __init__(self, filters: list[Any], transition_matrix: np.ndarray) -> None:
        self.filters = filters
        self.num_models = len(filters)
        self.transition_matrix = transition_matrix

        # Probabilities of each model
        self.mu_probs = np.ones(self.num_models) / self.num_models

        self.dim_x = filters[0].dim_x
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)

    @property
    def mu(self) -> np.ndarray:
        """Alias for mu_probs (backward compatibility)."""
        return self.mu_probs

    @mu.setter
    def mu(self, value: np.ndarray) -> None:
        self.mu_probs = value

    def predict(self, dt: float, **kwargs: Any) -> None:
        """
        Predict step (Mixing and model-specific prediction).

        Parameters
        ----------
        dt : float
            Time step (s).
        **kwargs : Any
            Additional arguments passed to the sub-filters' predict methods.
        """
        # Interaction (Mixing)
        # Calculate normalization constants cj = sum(Tij * mui)
        c_norm = self.mu_probs @ self.transition_matrix

        # Calculate mixing weights mu_ij = (Tij * mu_i) / cj
        mu_mixed = np.zeros((self.num_models, self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                mu_mixed[i, j] = (self.transition_matrix[i, j] * self.mu_probs[i]) / c_norm[j]

        # Mixed states and covariances
        x_mixed = [np.zeros(self.dim_x) for _ in range(self.num_models)]
        p_mixed = [np.zeros((self.dim_x, self.dim_x)) for _ in range(self.num_models)]

        # Mix states: x_mixed_j = sum(mu_ij * x_i)
        for j in range(self.num_models):
            for i in range(self.num_models):
                x_mixed[j] += mu_mixed[i, j] * self.filters[i].x

            for i in range(self.num_models):
                dx = self.filters[i].x - x_mixed[j]
                p_mixed[j] += mu_mixed[i, j] * (self.filters[i].P + np.outer(dx, dx))

        # Model-specific Prediction
        for i in range(self.num_models):
            self.filters[i].x = x_mixed[i]
            self.filters[i].P = p_mixed[i]

            # Support both 'fx' and 'fx_func' keywords
            fx_func = kwargs.get("fx_func", kwargs.get("fx"))

            sig = inspect.signature(self.filters[i].predict)
            if "fx_func" in sig.parameters and fx_func is not None:
                self.filters[i].predict(
                    dt=dt,
                    fx_func=fx_func,
                    **{k: v for k, v in kwargs.items() if k not in ["fx", "fx_func"]},
                )
            elif "dt" in sig.parameters:
                self.filters[i].predict(dt=dt, **kwargs)
            else:
                self.filters[i].predict(**kwargs)

    def update(self, z: np.ndarray, **kwargs: Any) -> None:
        """
        Update step (Model-specific update and mode probability update).

        Parameters
        ----------
        z : np.ndarray
            Measurement vector (dim_z,).
        **kwargs : Any
            Additional arguments passed to the sub-filters' update methods.
        """
        # Support both 'hx' and 'hx_func' keywords
        hx_func = kwargs.get("hx_func", kwargs.get("hx"))

        likelihoods = np.zeros(self.num_models)
        for i in range(self.num_models):
            if (
                "hx_func" in inspect.signature(self.filters[i].update).parameters
                and hx_func is not None
            ):
                self.filters[i].update(
                    z,
                    hx_func=hx_func,
                    **{k: v for k, v in kwargs.items() if k not in ["hx", "hx_func"]},
                )
            else:
                self.filters[i].update(z, **kwargs)
            likelihoods[i] = self._calculate_likelihood(i, z)

        # Mode probability update: mu_j = L_j * c_j / sum(L_k * c_k)
        c_norm = self.mu_probs @ self.transition_matrix
        self.mu_probs = likelihoods * c_norm
        self.mu_probs /= np.sum(self.mu_probs)

        # Combined state and covariance estimate
        self.x = np.zeros(self.dim_x)
        self.P = np.zeros((self.dim_x, self.dim_x))

        for i in range(self.num_models):
            self.x += self.mu_probs[i] * self.filters[i].x

        for i in range(self.num_models):
            dx = self.filters[i].x - self.x
            self.P += self.mu_probs[i] * (self.filters[i].P + np.outer(dx, dx))

    def _calculate_likelihood(self, model_idx: int, z: np.ndarray) -> float:
        """
        Calculate Gaussian measurement likelihood for a specific model.

        Parameters
        ----------
        model_idx : int
            Index of the model.
        z : np.ndarray
            Measurement vector.

        Returns
        -------
        float
            Likelihood value.
        """
        f = self.filters[model_idx]

        if hasattr(f, "H") and hasattr(f, "x"):
            h_mat = f.H
            resid = z - (h_mat @ f.x)
            s_mat = (h_mat @ f.P @ h_mat.T) + f.R
        else:
            # Fallback for filters that don't expose H (like UKF/CKF)
            return 1.0

        inv_s = np.linalg.inv(s_mat)
        det_s = np.linalg.det(s_mat)
        dim = len(z)
        exponent = -0.5 * (resid.T @ inv_s @ resid)
        return float((1.0 / np.sqrt((2 * np.pi) ** dim * det_s)) * np.exp(exponent))
