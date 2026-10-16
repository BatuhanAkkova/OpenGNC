"""
CCSDS Orbit Mean-Elements Message (OMM) implementation.
Standard: CCSDS 502.0-B-2 (Blue Book).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


class OMM:
    """
    Orbit Mean-Elements Message (OMM) handler.

    Supports reading, writing, and TLE cross-conversion of CCSDS OMM files (KVN format).
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.header: dict[str, str] = {
            "CCSDS_OMM_VERS": "2.0",
            "CREATION_DATE": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "ORIGINATOR": "OPENGNC",
        }
        self.metadata: dict[str, str] = {
            "OBJECT_NAME": "SPACECRAFT",
            "OBJECT_ID": "OPEN-GNC-01",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "TEME",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
        }
        self.mean_elements: dict[str, float | str] = {
            "EPOCH": "",
            "MEAN_MOTION": 0.0,  # rev/day
            "ECCENTRICITY": 0.0,
            "INCLINATION": 0.0,  # deg
            "RA_OF_ASC_NODE": 0.0,  # deg
            "ARG_OF_PERICENTER": 0.0,  # deg
            "MEAN_ANOMALY": 0.0,  # deg
        }
        self.sgp4_parameters: dict[str, float] = {
            "BSTAR": 0.0,  # 1/Earth-radii
            "MEAN_MOTION_DOT": 0.0,  # rev/day^2
            "MEAN_MOTION_DDOT": 0.0,  # rev/day^3
        }
        self.spacecraft_parameters: dict[str, float] = {
            "MASS": 100.0,  # kg
            "SOLAR_RAD_AREA": 1.0,  # m^2
            "SOLAR_RAD_COEFF": 1.2,
            "DRAG_AREA": 1.0,  # m^2
            "DRAG_COEFF": 2.2,
        }
        if data:
            self.set_data(data)

    def set_data(self, data: dict[str, Any]) -> None:
        """Update OMM fields from a dictionary."""
        for k, v in data.items():
            if k in self.header:
                self.header[k] = str(v)
            elif k in self.metadata:
                self.metadata[k] = str(v)
            elif k in self.mean_elements:
                self.mean_elements[k] = v if k == "EPOCH" else float(v)
            elif k in self.sgp4_parameters:
                self.sgp4_parameters[k] = float(v)
            elif k in self.spacecraft_parameters:
                self.spacecraft_parameters[k] = float(v)

    @classmethod
    def from_file(cls, filepath: str | Path) -> OMM:
        """Load an OMM from a KVN file."""
        omm = cls()
        content = Path(filepath).read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("COMMENT") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            clean_val = val.split("#")[0].split("[")[0].strip()

            if key in omm.header:
                omm.header[key] = clean_val
            elif key in omm.metadata:
                omm.metadata[key] = clean_val
            elif key in omm.mean_elements:
                omm.mean_elements[key] = clean_val if key == "EPOCH" else float(clean_val)
            elif key in omm.sgp4_parameters:
                omm.sgp4_parameters[key] = float(clean_val)
            elif key in omm.spacecraft_parameters:
                omm.spacecraft_parameters[key] = float(clean_val)
        return omm

    def to_string(self) -> str:
        """Render the OMM in standard Key-Value Notation (KVN)."""
        lines = [
            f"CCSDS_OMM_VERS = {self.header['CCSDS_OMM_VERS']}",
            f"CREATION_DATE  = {self.header['CREATION_DATE']}",
            f"ORIGINATOR     = {self.header['ORIGINATOR']}",
            "",
            "COMMENT OpenGNC Generated OMM",
            f"OBJECT_NAME          = {self.metadata['OBJECT_NAME']}",
            f"OBJECT_ID            = {self.metadata['OBJECT_ID']}",
            f"CENTER_NAME          = {self.metadata['CENTER_NAME']}",
            f"REF_FRAME            = {self.metadata['REF_FRAME']}",
            f"TIME_SYSTEM          = {self.metadata['TIME_SYSTEM']}",
            f"MEAN_ELEMENT_THEORY  = {self.metadata['MEAN_ELEMENT_THEORY']}",
            "",
            f"EPOCH                = {self.mean_elements['EPOCH']}",
            f"MEAN_MOTION          = {float(self.mean_elements['MEAN_MOTION']):.8f} [rev/day]",
            f"ECCENTRICITY         = {float(self.mean_elements['ECCENTRICITY']):.8f}",
            f"INCLINATION          = {float(self.mean_elements['INCLINATION']):.4f} [deg]",
            f"RA_OF_ASC_NODE       = {float(self.mean_elements['RA_OF_ASC_NODE']):.4f} [deg]",
            f"ARG_OF_PERICENTER    = {float(self.mean_elements['ARG_OF_PERICENTER']):.4f} [deg]",
            f"MEAN_ANOMALY         = {float(self.mean_elements['MEAN_ANOMALY']):.4f} [deg]",
            "",
            f"BSTAR                = {self.sgp4_parameters['BSTAR']:.8e} [1/ER]",
            f"MEAN_MOTION_DOT      = {self.sgp4_parameters['MEAN_MOTION_DOT']:.8e} [rev/day**2]",
            f"MEAN_MOTION_DDOT     = {self.sgp4_parameters['MEAN_MOTION_DDOT']:.8e} [rev/day**3]",
            "",
            f"MASS                 = {self.spacecraft_parameters['MASS']:.2f} [kg]",
            f"SOLAR_RAD_AREA       = {self.spacecraft_parameters['SOLAR_RAD_AREA']:.2f} [m**2]",
            f"SOLAR_RAD_COEFF      = {self.spacecraft_parameters['SOLAR_RAD_COEFF']:.2f}",
            f"DRAG_AREA            = {self.spacecraft_parameters['DRAG_AREA']:.2f} [m**2]",
            f"DRAG_COEFF           = {self.spacecraft_parameters['DRAG_COEFF']:.2f}",
        ]
        return "\n".join(lines)

    def to_file(self, filepath: str | Path) -> None:
        """Save the OMM to a KVN file."""
        Path(filepath).write_text(self.to_string(), encoding="utf-8")

    def write(self, filepath: str | Path) -> None:
        """Alias for to_file."""
        self.to_file(filepath)

    def get_keplerian_elements(self) -> dict[str, float]:
        """Return mean Keplerian orbital elements."""
        return {
            "mean_motion": float(self.mean_elements["MEAN_MOTION"]),
            "eccentricity": float(self.mean_elements["ECCENTRICITY"]),
            "inclination": float(self.mean_elements["INCLINATION"]),
            "raan": float(self.mean_elements["RA_OF_ASC_NODE"]),
            "arg_perigee": float(self.mean_elements["ARG_OF_PERICENTER"]),
            "mean_anomaly": float(self.mean_elements["MEAN_ANOMALY"]),
        }

    @classmethod
    def from_tle(cls, line1: str, line2: str, name: str = "OBJECT") -> OMM:
        """Create an OMM directly from a standard Two-Line Element (TLE) pair."""
        omm = cls()
        line1 = line1.strip()
        line2 = line2.strip()

        norad_id = line1[2:7].strip()
        intl_desig = line1[9:17].strip()

        epoch_yr = int(line1[18:20])
        full_year = 2000 + epoch_yr if epoch_yr < 57 else 1900 + epoch_yr
        epoch_day = float(line1[20:32])
        epoch_dt = datetime.datetime(
            full_year, 1, 1, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(days=epoch_day - 1.0)
        epoch_str = epoch_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        mean_motion_dot = float(line1[33:43].strip())
        bstar_mantissa = float(line1[53:59].strip()) / 1e5
        bstar_exp = int(line1[59:61].strip())
        bstar = bstar_mantissa * (10**bstar_exp)

        inc = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())
        ecc = float("0." + line2[26:33].strip())
        arg_p = float(line2[34:42].strip())
        m_anom = float(line2[43:51].strip())
        mm = float(line2[52:63].strip())

        omm.metadata["OBJECT_NAME"] = name
        omm.metadata["OBJECT_ID"] = f"{full_year}-{intl_desig}" if intl_desig else norad_id
        omm.mean_elements["EPOCH"] = epoch_str
        omm.mean_elements["MEAN_MOTION"] = mm
        omm.mean_elements["ECCENTRICITY"] = ecc
        omm.mean_elements["INCLINATION"] = inc
        omm.mean_elements["RA_OF_ASC_NODE"] = raan
        omm.mean_elements["ARG_OF_PERICENTER"] = arg_p
        omm.mean_elements["MEAN_ANOMALY"] = m_anom
        omm.sgp4_parameters["BSTAR"] = bstar
        omm.sgp4_parameters["MEAN_MOTION_DOT"] = mean_motion_dot
        return omm
