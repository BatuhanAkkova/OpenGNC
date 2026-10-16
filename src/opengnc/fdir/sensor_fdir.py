"""
Sensor Fault Detection, Isolation, and Characterization (FDIC) for spacecraft GNC.

Implements innovation-based Chi-Square statistical monitors, Generalized
Likelihood Ratio Test (GLRT) / CUSUM drift detectors, and analytical redundancy
cross-sensor isolation managers.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
from scipy import stats


class SensorHealth(Enum):
    """Health classification for spacecraft sensors."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    ISOLATED = "ISOLATED"


class ChiSquareMonitor:
    r"""
    Statistical Innovation Chi-Square ($\chi^2$) Fault Monitor.

    Evaluates the Normalized Innovation Squared (NIS):
    $$\gamma_k = \mathbf{r}_k^T \mathbf{S}_k^{-1} \mathbf{r}_k$$
    Under the null hypothesis $H_0$ (nominal sensor operation), $\gamma_k$
    follows a central chi-square distribution with $m$ degrees of freedom,
    where $m = \dim(\mathbf{r}_k)$.

    Parameters
    ----------
    dim : int
        Dimension of the measurement residual vector ($m$).
    confidence : float, optional
        Confidence level for the statistical hypothesis test (e.g., 0.99 for 1% $P_{FA}$).
        Default is 0.99.
    window_size : int, optional
        Sliding window size for cumulative hypothesis testing. Default is 5.
    persistence_threshold : int, optional
        Consecutive violations required before triggering a persistent alarm.
        Default is 3.
    """

    def __init__(
        self,
        dim: int,
        confidence: float = 0.99,
        window_size: int = 5,
        persistence_threshold: int = 3,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"Measurement dimension must be positive, got {dim}")
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        self.dim = dim
        self.confidence = confidence
        self.window_size = window_size
        self.persistence_threshold = persistence_threshold

        # Single-step threshold: chi2.ppf(confidence, m)
        self.threshold_single = float(stats.chi2.ppf(confidence, df=dim))
        # Windowed cumulative threshold: chi2.ppf(confidence, m * window_size)
        self.threshold_windowed = float(stats.chi2.ppf(confidence, df=dim * window_size))

        self.history: deque[float] = deque(maxlen=window_size)
        self.consecutive_violations: int = 0

    def update(self, residual: np.ndarray, innovation_cov: np.ndarray) -> tuple[bool, float, float]:
        r"""
        Update monitor with new innovation residual and covariance.

        Parameters
        ----------
        residual : np.ndarray
            Measurement innovation residual $\mathbf{r}_k = \mathbf{z}_k - \hat{\mathbf{z}}_k$.
        innovation_cov : np.ndarray
            Innovation covariance matrix $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k$.

        Returns
        -------
        tuple[bool, float, float]
            (is_fault, current_stat, threshold_used)
            - is_fault: True if fault condition detected.
            - current_stat: Computed statistical test metric ($\gamma_k$ or window sum).
            - threshold_used: Threshold corresponding to the test.
        """
        r = np.asarray(residual, dtype=float).flatten()
        S = np.asarray(innovation_cov, dtype=float)

        if len(r) != self.dim:
            raise ValueError(f"Expected residual of dimension {self.dim}, got {len(r)}")

        # Invert innovation covariance safely
        try:
            inv_S = np.linalg.pinv(S)
            nis = float(r.T @ inv_S @ r)
        except np.linalg.LinAlgError:
            return True, float("inf"), self.threshold_single

        self.history.append(nis)

        # Single-step spike detection
        single_violation = nis > self.threshold_single

        # Windowed cumulative detection
        windowed_violation = False
        window_sum = float(sum(self.history))
        if len(self.history) == self.window_size:
            windowed_violation = window_sum > self.threshold_windowed

        if single_violation or windowed_violation:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = max(0, self.consecutive_violations - 1)

        is_alarm = self.consecutive_violations >= self.persistence_threshold

        # Return test result
        active_stat = window_sum if (len(self.history) == self.window_size) else nis
        active_thresh = (
            self.threshold_windowed
            if (len(self.history) == self.window_size)
            else self.threshold_single
        )

        return is_alarm, active_stat, active_thresh

    def reset(self) -> None:
        """Reset internal history and violation counters."""
        self.history.clear()
        self.consecutive_violations = 0


class GLRTMonitor:
    r"""
    Generalized Likelihood Ratio Test (GLRT) / Page's CUSUM Drift Monitor.

    Detects insidious, low-amplitude sensor drift, temperature ramps, or slow
    random walks that stay below instantaneous $\chi^2$ single-step thresholds.

    Evaluates two-sided Cumulative Sum (CUSUM) statistics per channel:
    $$S_k^+ = \max(0, S_{k-1}^+ + r_k - \mu_0 - K)$$
    $$S_k^- = \max(0, S_{k-1}^- - r_k + \mu_0 - K)$$

    Parameters
    ----------
    dim : int
        Dimension of the monitored signal.
    drift_allowance : float, optional
        Allowance constant $K$ (slack parameter). Default is 0.5.
    alarm_threshold : float, optional
        Decision threshold $H$. Default is 5.0.
    nominal_mean : float, optional
        Expected nominal mean $\mu_0$ under $H_0$. Default is 0.0.
    """

    def __init__(
        self,
        dim: int,
        drift_allowance: float = 0.5,
        alarm_threshold: float = 5.0,
        nominal_mean: float = 0.0,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if drift_allowance < 0.0:
            raise ValueError(f"Drift allowance must be non-negative, got {drift_allowance}")
        if alarm_threshold <= 0.0:
            raise ValueError(f"Alarm threshold must be positive, got {alarm_threshold}")

        self.dim = dim
        self.k = float(drift_allowance)
        self.h = float(alarm_threshold)
        self.mu0 = float(nominal_mean)

        self.s_pos: np.ndarray = np.zeros(dim, dtype=float)
        self.s_neg: np.ndarray = np.zeros(dim, dtype=float)

    def update(self, residual: np.ndarray | float) -> tuple[bool, np.ndarray]:
        """
        Update CUSUM statistics with a new measurement residual.

        Parameters
        ----------
        residual : np.ndarray | float
            Measurement residual or scalar.

        Returns
        -------
        tuple[bool, np.ndarray]
            (is_fault, max_cusum_stat)
            - is_fault: True if any channel exceeds the decision threshold $H$.
            - max_cusum_stat: Maximum cumulative statistic across positive/negative detectors.
        """
        r = np.asarray(residual, dtype=float).flatten()
        if len(r) != self.dim:
            raise ValueError(f"Expected residual of length {self.dim}, got {len(r)}")

        self.s_pos = np.asarray(np.maximum(0.0, self.s_pos + (r - self.mu0) - self.k), dtype=float)
        self.s_neg = np.asarray(np.maximum(0.0, self.s_neg - (r - self.mu0) - self.k), dtype=float)

        max_stat = np.maximum(self.s_pos, self.s_neg)
        is_fault = bool(np.any(max_stat > self.h))

        return is_fault, max_stat

    def reset(self) -> None:
        """Reset cumulative sum registers."""
        self.s_pos.fill(0.0)
        self.s_neg.fill(0.0)


class SensorIsolationManager:
    r"""
    Analytical Redundancy and Cross-Sensor Isolation Manager.

    Coordinates health tracking across redundant sensors, performs pairwise
    consistency voting to identify out-of-family sensors, and provides dynamic
    measurement covariance inflation ($R_{eff} = s \cdot R$) to prevent filter
    divergence before hard isolation.

    Parameters
    ----------
    sensor_names : list[str]
        List of unique sensor identifiers.
    max_consecutive_faults : int, optional
        Consecutive faults before a sensor is isolated. Default is 5.
    covariance_inflation_factor : float, optional
        Scaling factor $s$ applied to degraded sensor covariance. Default is 10.0.
    """

    def __init__(
        self,
        sensor_names: list[str],
        max_consecutive_faults: int = 5,
        covariance_inflation_factor: float = 10.0,
    ) -> None:
        if not sensor_names:
            raise ValueError("Must provide at least one sensor name")

        self.sensors: dict[str, SensorHealth] = {
            name: SensorHealth.HEALTHY for name in sensor_names
        }
        self.fault_counters: dict[str, int] = {name: 0 for name in sensor_names}
        self.max_faults = max_consecutive_faults
        self.inflation_factor = float(covariance_inflation_factor)

    def report_status(self, sensor_name: str, has_fault: bool) -> SensorHealth:
        """
        Report detection result for a given sensor and update health state.

        Parameters
        ----------
        sensor_name : str
            Identifier of the reporting sensor.
        has_fault : bool
            True if FDI monitor detected an anomaly on this sensor.

        Returns
        -------
        SensorHealth
            Updated health status of the sensor.
        """
        if sensor_name not in self.sensors:
            raise KeyError(f"Sensor '{sensor_name}' not registered in manager")

        if self.sensors[sensor_name] == SensorHealth.ISOLATED:
            return SensorHealth.ISOLATED

        if has_fault:
            self.fault_counters[sensor_name] += 1
            if self.fault_counters[sensor_name] >= self.max_faults:
                self.sensors[sensor_name] = SensorHealth.ISOLATED
            else:
                self.sensors[sensor_name] = SensorHealth.DEGRADED
        else:
            self.fault_counters[sensor_name] = max(0, self.fault_counters[sensor_name] - 1)
            if self.fault_counters[sensor_name] == 0:
                self.sensors[sensor_name] = SensorHealth.HEALTHY

        return self.sensors[sensor_name]

    def get_effective_covariance(self, sensor_name: str, base_cov: np.ndarray) -> np.ndarray:
        r"""
        Calculate effective covariance matrix $R_{eff}$ for the sensor.

        For healthy sensors, returns $R$. For degraded sensors, returns
        $s \cdot R$. For isolated sensors, returns $\infty \cdot I$.

        Parameters
        ----------
        sensor_name : str
            Identifier of the sensor.
        base_cov : np.ndarray
            Nominal measurement covariance matrix $R$.

        Returns
        -------
        np.ndarray
            Adjusted effective covariance matrix.
        """
        state = self.sensors[sensor_name]
        cov = np.asarray(base_cov, dtype=float)

        if state == SensorHealth.HEALTHY:
            return cov.copy()
        elif state == SensorHealth.DEGRADED:
            return cov * self.inflation_factor
        else:
            # Isolated: effectively infinite covariance (decoupled)
            return cov * 1e9

    def isolate_sensor(self, sensor_name: str) -> None:
        """Manually force a sensor into the ISOLATED state."""
        if sensor_name in self.sensors:
            self.sensors[sensor_name] = SensorHealth.ISOLATED
            self.fault_counters[sensor_name] = self.max_faults

    def reset_sensor(self, sensor_name: str) -> None:
        """Manually restore a sensor to HEALTHY state."""
        if sensor_name in self.sensors:
            self.sensors[sensor_name] = SensorHealth.HEALTHY
            self.fault_counters[sensor_name] = 0

    def cross_consistency_vote(
        self, measurements: dict[str, np.ndarray], angular_tol_deg: float = 2.0
    ) -> list[str]:
        """
        Perform 2-out-of-N cross-consistency voting among vector sensors.

        Identifies sensors whose output diverges significantly from the consensus.

        Parameters
        ----------
        measurements : dict[str, np.ndarray]
            Dictionary of unit vector measurements {sensor_name: unit_vector}.
        angular_tol_deg : float, optional
            Maximum allowable angular divergence in degrees. Default is 2.0.

        Returns
        -------
        list[str]
            List of suspect sensor names that failed consensus voting.
        """
        names = [n for n in measurements if self.sensors.get(n) != SensorHealth.ISOLATED]
        if len(names) < 3:
            return []  # Need at least 3 active sensors for majority consensus voting

        cos_tol = np.cos(np.radians(angular_tol_deg))
        divergence_count: dict[str, int] = {n: 0 for n in names}

        for i, name1 in enumerate(names):
            v1 = measurements[name1] / np.linalg.norm(measurements[name1])
            for name2 in names[i + 1 :]:
                v2 = measurements[name2] / np.linalg.norm(measurements[name2])
                cos_angle = float(np.dot(v1, v2))
                if cos_angle < cos_tol:
                    divergence_count[name1] += 1
                    divergence_count[name2] += 1

        # Outlier is the sensor with majority disagreement
        outliers = [n for n, count in divergence_count.items() if count >= (len(names) - 1)]
        for outlier in outliers:
            self.report_status(outlier, has_fault=True)

        return outliers
