import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from opengnc.simulation.scenario import ScenarioConfig


def test_scenario_config_json(tmp_path):
    config_file = tmp_path / "scenario.json"
    data = {"sat": {"mass": 500.0, "name": "Hawk"}}
    with open(config_file, "w") as f:
        json.dump(data, f)

    cfg = ScenarioConfig(config_file)
    assert cfg.get("sat.mass") == 500.0
    assert cfg.get("sat.name") == "Hawk"
    assert cfg.get("sat.invalid") is None
    assert cfg.get("sat.mass.sub") is None


def test_scenario_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        ScenarioConfig("non_existent_file.json")


def test_scenario_config_unsupported_ext(tmp_path):
    config_file = tmp_path / "scenario.txt"
    config_file.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported configuration"):
        ScenarioConfig(config_file)


def test_scenario_config_yaml_import_error(tmp_path):
    config_file = tmp_path / "scenario.yaml"
    config_file.write_text("sat:\n  mass: 500.0")

    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="PyYAML is required"):
            ScenarioConfig(config_file)


def test_scenario_config_yaml_success(tmp_path):
    config_file = tmp_path / "scenario.yaml"
    config_file.write_text("sat:\n  mass: 500.0")

    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"sat": {"mass": 500.0}}
    with patch.dict("sys.modules", {"yaml": mock_yaml}):
        cfg = ScenarioConfig(config_file)
        assert cfg.get("sat.mass") == 500.0
