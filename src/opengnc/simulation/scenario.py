"""
Scenario Configuration Manager.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class ScenarioConfig:
    """
    Scenario Configuration Manager.

    Handles loading and parsing of reproducible mission configurations from
    external formatted files (JSON, YAML).

    Parameters
    ----------
    filename : Union[str, Path]
        Path to the configuration file.
    """

    def __init__(self, filename: str | Path) -> None:
        """Initialize and automatically load configuration."""
        self.filename = Path(filename)
        self.config: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        Load data from the filesystem into the internal configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        ImportError
            If YAML is requested but PyYAML is missing.
        ValueError
            If file extension is unsupported.
        """
        if not self.filename.exists():
            raise FileNotFoundError(f"Scenario configuration file not found: {self.filename}")

        ext = self.filename.suffix.lower()
        if ext == ".json":
            with open(self.filename, encoding="utf-8") as f:
                self.config = json.load(f)
        elif ext in [".yaml", ".yml"]:
            try:
                import yaml  # type: ignore[import-untyped]
                with open(self.filename, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.config = cast(dict[str, Any], data) if data is not None else {}
            except ImportError:
                raise ImportError(
                    "PyYAML is required to parse YAML scenarios. Install it with `pip install pyyaml`."
                )
        else:
            raise ValueError(f"Unsupported configuration file extension: {ext}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot-notation.

        Example: `config.get('spacecraft.mass', 500.0)` retrieves the 'mass'
        property from the 'spacecraft' object.

        Parameters
        ----------
        key : str
            Dot-separated access path (e.g., 'propulsion.tank_vol').
        default : Any, optional
            Fallback value if key is missing.

        Returns
        -------
        Any
            The requested parameter value.
        """
        keys = key.split(".")
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current




