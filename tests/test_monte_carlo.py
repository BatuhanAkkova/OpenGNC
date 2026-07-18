from unittest.mock import MagicMock, patch

from opengnc.simulation.monte_carlo import MonteCarloSim


class DummySimulator:
    def __init__(self, seed: int, **kwargs: str) -> None:
        self.seed = seed
        self.kwargs = kwargs

    def run(self) -> dict[str, int | str | None]:
        return {"seed": self.seed, "param": self.kwargs.get("param")}


def simulator_factory(seed: int, **kwargs: str) -> DummySimulator:
    return DummySimulator(seed, **kwargs)


def test_monte_carlo_sequential() -> None:
    mc = MonteCarloSim(simulator_factory)
    mc.run_sequential(num_runs=3, param="test")

    assert len(mc.results) == 3
    for i in range(3):
        assert mc.results[i] == {"seed": i, "param": "test"}


@patch("opengnc.simulation.monte_carlo.mp.Pool")
def test_monte_carlo_parallel(mock_pool_class: MagicMock) -> None:
    mock_pool = MagicMock()
    mock_pool_class.return_value.__enter__.return_value = mock_pool

    mc = MonteCarloSim(simulator_factory)

    def mock_map(func, iterable):
        return [func(x) for x in iterable]

    mock_pool.map.side_effect = mock_map

    with patch("joblib.Parallel", side_effect=PermissionError):
        mc.run_parallel(num_runs=4, processes=2, param="parallel")

    assert len(mc.results) == 4
    results = sorted(mc.results, key=lambda x: x["seed"])
    for i in range(4):
        assert results[i] == {"seed": i, "param": "parallel"}


def functional_factory(seed: int, **kwargs: str) -> dict[str, int | str | None]:
    return {"seed": seed, "param": kwargs.get("param")}


def test_monte_carlo_functional() -> None:
    mc = MonteCarloSim(functional_factory)
    mc.run_sequential(num_runs=2, param="func")

    assert len(mc.results) == 2
    assert mc.results[0] == {"seed": 0, "param": "func"}
    assert mc.results[1] == {"seed": 1, "param": "func"}


@patch("opengnc.simulation.monte_carlo.mp.Pool")
def test_monte_carlo_parallel_functional(mock_pool_class: MagicMock) -> None:
    mock_pool = MagicMock()
    mock_pool_class.return_value.__enter__.return_value = mock_pool

    mc = MonteCarloSim(functional_factory)

    def mock_map(func, iterable):
        return [func(x) for x in iterable]

    mock_pool.map.side_effect = mock_map

    with patch("joblib.Parallel", side_effect=PermissionError):
        mc.run_parallel(num_runs=2, param="parallel_func")

    assert len(mc.results) == 2
    assert mc.results[0] == {"seed": 0, "param": "parallel_func"}
