"""
CCSDS Orbit Ephemeris Message (OEM) implementation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


class OEM:
    """
    Orbit Ephemeris Message (OEM) handler.

    Supports reading and writing CCSDS OEM files.
    """

    def __init__(self) -> None:
        self.header = {
            "CCSDS_OEM_VERS": "2.0",
            "CREATION_DATE": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "ORIGINATOR": "OPENGNC",
        }
        self.metadata = {
            "OBJECT_NAME": "SPACECRAFT",
            "OBJECT_ID": "OPEN-GNC-01",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "EME2000",
            "TIME_SYSTEM": "UTC",
            "INTERPOLATION": "LAGRANGE",
            "INTERPOLATION_DEGREE": "7",
        }
        self.data: pd.DataFrame | None = None

    def set_data(self, epochs: list[str], states: np.ndarray) -> None:
        """Set ephemeris data."""
        self.data = pd.DataFrame(
            np.asarray(states, dtype=float),
            columns=["X", "Y", "Z", "VX", "VY", "VZ"],
            index=epochs,
        )
        self.metadata["START_TIME"] = epochs[0]
        self.metadata["STOP_TIME"] = epochs[-1]

    @classmethod
    def from_file(cls, filepath: str) -> OEM:
        """Load an OEM from a file."""
        oem = cls()
        metadata_mode = False
        data_rows: list[list[float | str]] = []

        for line in Path(filepath).read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("COMMENT"):
                continue
            if stripped == "META_START":
                metadata_mode = True
                continue
            if stripped == "META_STOP":
                metadata_mode = False
                continue

            if "=" in stripped:
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.split("#")[0].strip()
                if metadata_mode:
                    oem.metadata[key] = val
                elif key in oem.header:
                    oem.header[key] = val
            else:
                parts = stripped.split()
                if len(parts) >= 7:
                    epoch = parts[0]
                    state = [float(p) for p in parts[1:7]]
                    data_rows.append([epoch, *state])

        if data_rows:
            oem.data = pd.DataFrame(
                [row[1:] for row in data_rows],
                columns=["X", "Y", "Z", "VX", "VY", "VZ"],
                index=[cast(str, row[0]) for row in data_rows],
            )
        return oem

    def write(self, filepath: str) -> None:
        """Write the OEM to a file."""
        if self.data is None:
            raise ValueError("No ephemeris data to write.")

        output = Path(filepath)
        with output.open("w", encoding="utf-8") as f_handle:
            f_handle.write(f"CCSDS_OEM_VERS = {self.header['CCSDS_OEM_VERS']}\n")
            f_handle.write(f"CREATION_DATE  = {self.header['CREATION_DATE']}\n")
            f_handle.write(f"ORIGINATOR     = {self.header['ORIGINATOR']}\n\n")

            f_handle.write("META_START\n")
            for key, value in self.metadata.items():
                f_handle.write(f"{key} = {value}\n")
            f_handle.write("META_STOP\n\n")

            for epoch, row in self.data.iterrows():
                f_handle.write(
                    f"{epoch} {row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f} "
                    f"{row['VX']:.9f} {row['VY']:.9f} {row['VZ']:.9f}\n"
                )

    def get_interpolated_state(self, epoch: str) -> np.ndarray:
        """
        Placeholder for orbital interpolation logic.

        Returns the exact stored state for now.
        """
        if self.data is not None and epoch in self.data.index:
            row = self.data.loc[epoch]
            return cast(np.ndarray, np.asarray(row.to_numpy(dtype=float) * 1000.0, dtype=float))
        raise NotImplementedError("High-fidelity interpolation not yet implemented.")
