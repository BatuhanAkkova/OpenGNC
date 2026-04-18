"""
CCSDS Orbit Ephemeris Message (OEM) implementation.
"""

import datetime
from typing import List, Optional

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
        self.data: Optional[pd.DataFrame] = None

    def set_data(self, epochs: List[str], states: np.ndarray) -> None:
        """
        Set ephemeris data.

        Parameters
        ----------
        epochs : List[str]
            List of ISO-8601 strings.
        states : np.ndarray
            Nx6 array of states [x, y, z, vx, vy, vz] in km and km/s.
        """
        self.data = pd.DataFrame(
            states, columns=["X", "Y", "Z", "VX", "VY", "VZ"], index=epochs
        )
        self.metadata["START_TIME"] = epochs[0]
        self.metadata["STOP_TIME"] = epochs[-1]

    @classmethod
    def from_file(cls, filepath: str) -> "OEM":
        """Load OEM from a file."""
        oem = cls()
        metadata_mode = False
        data_rows = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("COMMENT"):
                    continue
                if line == "META_START":
                    metadata_mode = True
                    continue
                if line == "META_STOP":
                    metadata_mode = False
                    continue

                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.split("#")[0].strip()
                    if metadata_mode:
                        oem.metadata[key] = val
                    elif key in oem.header:
                        oem.header[key] = val
                else:
                    # Data line
                    parts = line.split()
                    if len(parts) >= 7:
                        epoch = parts[0]
                        state = [float(p) for p in parts[1:7]]
                        data_rows.append([epoch] + state)

        if data_rows:
            df = pd.DataFrame(
                [r[1:] for r in data_rows],
                columns=["X", "Y", "Z", "VX", "VY", "VZ"],
                index=[r[0] for r in data_rows],
            )
            oem.data = df
        return oem

    def write(self, filepath: str) -> None:
        """Write OEM to a file."""
        if self.data is None:
            raise ValueError("No ephemeris data to write.")

        with open(filepath, "w") as f:
            f.write(f"CCSDS_OEM_VERS = {self.header['CCSDS_OEM_VERS']}\n")
            f.write(f"CREATION_DATE  = {self.header['CREATION_DATE']}\n")
            f.write(f"ORIGINATOR     = {self.header['ORIGINATOR']}\n\n")

            f.write("META_START\n")
            for k, v in self.metadata.items():
                f.write(f"{k} = {v}\n")
            f.write("META_STOP\n\n")

            for epoch, row in self.data.iterrows():
                f.write(
                    f"{epoch} {row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f} "
                    f"{row['VX']:.9f} {row['VY']:.9f} {row['VZ']:.9f}\n"
                )

    def get_interpolated_state(self, epoch: str) -> np.ndarray:
        """
        Placeholder for orbital interpolation logic.
        In Phase 2, this would use Lagrange/Hermite interpolation.
        """
        # For now, return nearest neighbor if exact match, else raise
        if self.data is not None and epoch in self.data.index:
            row = self.data.loc[epoch]
            return row.values * 1000.0  # Convert to SI (m, m/s)
        raise NotImplementedError("High-fidelity interpolation not yet implemented.")
