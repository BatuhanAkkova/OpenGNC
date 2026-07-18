from collections.abc import Callable
from typing import Any

import numpy as np


class MonteCarloAnalyzer:
    """
    Statistical analysis suite for Monte Carlo simulations.

    Provides tools for calculating performance margins, stability metrics,
    and covariance consistency checks.
    """

    def __init__(self, results: list[dict[str, Any]]) -> None:
        """Initialize with a list of simulation result dictionaries."""
        self.results = results
        if not results:
            raise ValueError("No results provided to MonteCarloAnalyzer.")

    def get_aggregate_stats(self, key: str) -> dict[str, np.ndarray]:
        """
        Extract summary statistics for a specific telemetry key.

        Parameters
        ----------
        key : str
            Result dictionary key containing scalar or array-like telemetry.

        Returns
        -------
        dict[str, np.ndarray]
            Mean, spread, and extrema statistics for the requested key.
        """
        data = [res[key] for res in self.results if key in res]
        if not data:
            return {}

        arr = np.asarray(data, dtype=float)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)

        return {
            "mean": mean,
            "std": std,
            "sigma_3_upper": mean + 3.0 * std,
            "sigma_3_lower": mean - 3.0 * std,
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
            "median": np.median(arr, axis=0),
        }

    def check_covariance_consistency(self, error_key: str, cov_key: str) -> dict[str, float | str]:
        """
        Perform a covariance consistency check such as NIS or NEES.

        Parameters
        ----------
        error_key : str
            Result key for estimation error vectors.
        cov_key : str
            Result key for covariance matrices.

        Returns
        -------
        dict[str, float | str]
            Aggregate consistency metrics, or ``{"status": "missing_data"}``
            when the required keys are not present.
        """
        errors = [
            np.asarray(res[error_key], dtype=float) for res in self.results if error_key in res
        ]
        covariances = [
            np.asarray(res[cov_key], dtype=float) for res in self.results if cov_key in res
        ]

        if not errors or not covariances:
            return {"status": "missing_data"}

        nis_values: list[float] = []
        for error, covariance in zip(errors, covariances):
            if error.ndim > 1:
                run_nis: list[float] = []
                for i in range(len(error)):
                    try:
                        inv_cov = np.linalg.inv(covariance[i])
                    except np.linalg.LinAlgError:
                        continue
                    run_nis.append(float(error[i].T @ inv_cov @ error[i]))
                if run_nis:
                    nis_values.append(float(np.mean(run_nis)))
            else:
                try:
                    inv_cov = np.linalg.inv(covariance)
                except np.linalg.LinAlgError:
                    continue
                nis_values.append(float(error.T @ inv_cov @ error))

        if not nis_values:
            return {"status": "missing_data"}

        expected_dim = float(errors[0].shape[-1])
        avg_nis = float(np.mean(nis_values))
        return {
            "avg_nis": avg_nis,
            "expected": expected_dim,
            "consistency_ratio": avg_nis / expected_dim,
        }

    def summarize_failures(
        self, criteria_func: Callable[[dict[str, Any]], bool]
    ) -> dict[str, int | float]:
        """
        Calculate failure rates based on a user-defined criteria function."""
        failures = sum(1 for res in self.results if criteria_func(res))
        rate = failures / len(self.results)
        return {
            "total_runs": len(self.results),
            "failures": failures,
            "failure_rate": rate,
            "reliability": 1.0 - rate,
        }
