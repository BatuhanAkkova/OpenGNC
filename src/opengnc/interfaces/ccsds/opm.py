"""
CCSDS Orbit Parameter Message (OPM) implementation.
"""

import datetime

import numpy as np


class OPM:
    """
    Orbit Parameter Message (OPM) handler.

    Supports reading and writing CCSDS OPM files (KVN format).
    """

    def __init__(self, data: dict | None = None) -> None:
        self.header = {
            "CCSDS_OPM_VERS": "2.0",
            "CREATION_DATE": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "ORIGINATOR": "OPENGNC",
        }
        self.metadata = {
            "OBJECT_NAME": "SPACECRAFT",
            "OBJECT_ID": "OPEN-GNC-01",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "EME2000",
            "TIME_SYSTEM": "UTC",
        }
        self.state: dict[str, float | str] = {
            "EPOCH": "",
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "X_DOT": 0.0,
            "Y_DOT": 0.0,
            "Z_DOT": 0.0,
        }
        if data:
            self.state.update(data)

    @classmethod
    def from_file(cls, filepath: str) -> "OPM":
        """Load OPM from a KVN file."""
        opm = cls()
        with open(filepath) as f:
            for line in f:
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.split("#")[0].strip()  # Remove comments
                    # Handle units in brackets, e.g., "6678.14 [km]"
                    clean_val = val.split("[")[0].strip()
                    if key in opm.header:
                        opm.header[key] = val
                    elif key in opm.metadata:
                        opm.metadata[key] = val
                    elif key in opm.state:
                        try:
                            opm.state[key] = float(clean_val)
                        except ValueError:
                            opm.state[key] = val
        return opm

    def write(self, filepath: str) -> None:
        """Write OPM to a KVN file."""
        with open(filepath, "w") as f:
            f.write(f"CCSDS_OPM_VERS = {self.header['CCSDS_OPM_VERS']}\n")
            f.write(f"CREATION_DATE  = {self.header['CREATION_DATE']}\n")
            f.write(f"ORIGINATOR     = {self.header['ORIGINATOR']}\n\n")

            f.write("COMMENT Metadata\n")
            for k, v in self.metadata.items():
                f.write(f"{k} = {v}\n")

            f.write("\nCOMMENT State Vector\n")
            f.write(f"EPOCH = {self.state['EPOCH']}\n")
            f.write(f"X     = {self.state['X']:.6f} [km]\n")
            f.write(f"Y     = {self.state['Y']:.6f} [km]\n")
            f.write(f"Z     = {self.state['Z']:.6f} [km]\n")
            f.write(f"X_DOT = {self.state['X_DOT']:.9f} [km/s]\n")
            f.write(f"Y_DOT = {self.state['Y_DOT']:.9f} [km/s]\n")
            f.write(f"Z_DOT = {self.state['Z_DOT']:.9f} [km/s]\n")

    def get_state_vector(self) -> np.ndarray:
        """Return state vector in SI units (m, m/s)."""
        return np.array(
            [
                float(self.state["X"]) * 1000.0,
                float(self.state["Y"]) * 1000.0,
                float(self.state["Z"]) * 1000.0,
                float(self.state["X_DOT"]) * 1000.0,
                float(self.state["Y_DOT"]) * 1000.0,
                float(self.state["Z_DOT"]) * 1000.0,
            ]
        )
