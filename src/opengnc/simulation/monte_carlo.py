"""
Monte Carlo simulation harness and statistical analysis helpers.
"""

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

from .monte_carlo_analyzer import MonteCarloAnalyzer


class MonteCarloSim:
    """
    Monte Carlo simulation harness.

    Facilitates large-scale robustness analysis by executing multiple
    simulation runs with stochastic variations.

    Parameters
    ----------
    simulator_factory : Callable[..., Any]
        Generator function that produces simulator instances or direct results.
        Signature: ``(seed, **kwargs) -> simulator_or_result``.
    """

    def __init__(self, simulator_factory: Callable[..., Any]) -> None:
        """Initialize the harness with a simulator factory."""
        self.simulator_factory = simulator_factory
        self.results: list[Any] = []

    def _run_single(self, kwargs: dict[str, Any]) -> Any:
        """Execute a single Monte Carlo trial."""
        params = dict(kwargs)
        seed = params.pop("seed")
        sim = self.simulator_factory(seed, **params)
        if hasattr(sim, "run") and callable(sim.run):
            return sim.run()
        return sim

    def run_sequential(self, num_runs: int, **kwargs: Any) -> list[Any]:
        """
        Execute Monte Carlo iterations in a single thread.

        Parameters
        ----------
        num_runs : int
            Number of trials to execute.
        **kwargs : Any
            Variable parameters passed to the simulator factory.

        Returns
        -------
        list[Any]
            Aggregated results from all trials.
        """
        self.results = []
        for i in range(num_runs):
            params = dict(kwargs)
            params["seed"] = i
            self.results.append(self._run_single(params))
        return self.results

    def run_parallel(
        self,
        num_runs: int,
        processes: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Execute Monte Carlo iterations across multiple processor cores.

        Parameters
        ----------
        num_runs : int
            Number of trials to execute.
        processes : int | None, optional
            Number of parallel workers. Defaults to machine CPU count.
        **kwargs : Any
            Parameters for simulation configuration.

        Returns
        -------
        list[Any]
            Aggregated results.
        """
        self.results = []
        pool_kwargs: list[dict[str, Any]] = []
        for i in range(num_runs):
            params = dict(kwargs)
            params["seed"] = i
            pool_kwargs.append(params)

        if processes == 1:
            return self.run_sequential(num_runs=num_runs, **kwargs)

        try:
            from joblib import Parallel, delayed

            self.results = Parallel(n_jobs=processes, backend="loky")(
                delayed(self._run_single)(p) for p in pool_kwargs
            )
            return self.results
        except (ImportError, OSError, PermissionError):
            pass

        try:
            with mp.Pool(processes) as pool:
                self.results = pool.map(self._run_single, pool_kwargs)
            return self.results
        except (OSError, PermissionError):
            return self.run_sequential(num_runs=num_runs, **kwargs)

    def get_analyzer(self) -> MonteCarloAnalyzer:
        """
        Produce a statistical analyzer for the current simulation results.

        Returns
        -------
        MonteCarloAnalyzer
            Analyzer initialized with the current trial data.
        """
        return MonteCarloAnalyzer(self.results)
