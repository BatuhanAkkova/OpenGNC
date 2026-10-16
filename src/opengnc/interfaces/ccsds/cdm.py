"""
CCSDS Conjunction Data Message (CDM) implementation.
Standard: CCSDS 508.0-B-1 (Blue Book).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np


class CDM:
    """
    Conjunction Data Message (CDM) handler.

    Supports reading, writing, and analytical collision probability calculation ($P_c$)
    per CCSDS 508.0-B-1 and Foster (1992) standards.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.header: dict[str, str] = {
            "CCSDS_CDM_VERS": "1.0",
            "CREATION_DATE": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "ORIGINATOR": "OPENGNC",
            "MESSAGE_FOR": "SATELLITE_OPERATOR",
            "MESSAGE_ID": "OPENGNC-CDM-001",
        }
        self.relative_metadata: dict[str, Any] = {
            "TCA": "",
            "MISS_DISTANCE": 0.0,  # m
            "RELATIVE_SPEED": 0.0,  # m/s
            "RELATIVE_POSITION_R": 0.0,  # m
            "RELATIVE_POSITION_T": 0.0,  # m
            "RELATIVE_POSITION_N": 0.0,  # m
            "COLLISION_PROBABILITY": 0.0,
            "COLLISION_PROB_METHOD": "FOSTER-1992",
        }
        self.object1: dict[str, Any] = {
            "OBJECT": "OBJECT1",
            "OBJECT_DESIGNATOR": "SAT-01",
            "CATALOG_NAME": "SATCAT",
            "REF_FRAME": "RTN",
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "X_DOT": 0.0,
            "Y_DOT": 0.0,
            "Z_DOT": 0.0,
            "HARD_BODY_RADIUS": 2.0,  # m
            "COV_RTN": np.diag([100.0, 500.0, 100.0, 0.01, 0.05, 0.01]),  # m^2, (m/s)^2
        }
        self.object2: dict[str, Any] = {
            "OBJECT": "OBJECT2",
            "OBJECT_DESIGNATOR": "DEBRIS-01",
            "CATALOG_NAME": "SATCAT",
            "REF_FRAME": "RTN",
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "X_DOT": 0.0,
            "Y_DOT": 0.0,
            "Z_DOT": 0.0,
            "HARD_BODY_RADIUS": 1.0,  # m
            "COV_RTN": np.diag([200.0, 1000.0, 200.0, 0.02, 0.1, 0.02]),
        }
        if data:
            self.set_data(data)

    def set_data(self, data: dict[str, Any]) -> None:
        """Update CDM fields from a dictionary."""
        for k, v in data.items():
            if k in self.header:
                self.header[k] = str(v)
            elif k in self.relative_metadata:
                self.relative_metadata[k] = v
            elif k.startswith("OBJ1_"):
                self.object1[k[5:]] = v
            elif k.startswith("OBJ2_"):
                self.object2[k[5:]] = v

    def compute_collision_probability(self, hard_body_radius: float | None = None) -> float:
        r"""
        Compute the 2D collision probability ($P_c$) in the encounter frame (Foster, 1992).

        Parameters
        ----------
        hard_body_radius : float, optional
            Combined spherical hard-body radius $R = R_1 + R_2$ (m).
            If omitted, uses the sum of `object1` and `object2` radii.

        Returns
        -------
        float
            Collision probability $P_c \in [0, 1]$.
        """
        if hard_body_radius is None:
            r1 = float(self.object1.get("HARD_BODY_RADIUS", 2.0))
            r2 = float(self.object2.get("HARD_BODY_RADIUS", 1.0))
            radius = r1 + r2
        else:
            radius = float(hard_body_radius)

        cov1_pos = np.asarray(self.object1["COV_RTN"][:3, :3], dtype=float)
        cov2_pos = np.asarray(self.object2["COV_RTN"][:3, :3], dtype=float)
        comb_cov = cov1_pos + cov2_pos

        # Encounter plane: miss vector in RTN (projected along R and N)
        miss_r = float(self.relative_metadata.get("RELATIVE_POSITION_R", 0.0))
        miss_n = float(self.relative_metadata.get("RELATIVE_POSITION_N", 0.0))

        # 2D combined covariance in the (R, N) encounter projection
        cov_2d = np.array(
            [
                [comb_cov[0, 0], comb_cov[0, 2]],
                [comb_cov[2, 0], comb_cov[2, 2]],
            ]
        )
        det_cov = float(np.linalg.det(cov_2d))
        if det_cov <= 1e-14:
            return 0.0

        inv_cov = np.linalg.inv(cov_2d)
        d_vec = np.array([miss_r, miss_n])
        mahalanobis_sq = float(d_vec.T @ inv_cov @ d_vec)

        # Standard Akella-Alfriend / Chan series approximation for R^2 << det(cov)
        p_c = (radius**2 / (2.0 * np.sqrt(det_cov))) * np.exp(-0.5 * mahalanobis_sq)
        p_c = float(np.clip(p_c, 0.0, 1.0))
        self.relative_metadata["COLLISION_PROBABILITY"] = p_c
        return p_c

    def to_string(self) -> str:
        """Format CDM in standard CCSDS Key-Value Notation (KVN)."""
        lines = [
            f"CCSDS_CDM_VERS = {self.header['CCSDS_CDM_VERS']}",
            f"CREATION_DATE  = {self.header['CREATION_DATE']}",
            f"ORIGINATOR     = {self.header['ORIGINATOR']}",
            f"MESSAGE_FOR    = {self.header['MESSAGE_FOR']}",
            f"MESSAGE_ID     = {self.header['MESSAGE_ID']}",
            "",
            "COMMENT Conjunction Assessment Metadata",
            f"TCA                   = {self.relative_metadata['TCA']}",
            f"MISS_DISTANCE         = {float(self.relative_metadata['MISS_DISTANCE']):.3f} [m]",
            f"RELATIVE_SPEED        = {float(self.relative_metadata['RELATIVE_SPEED']):.3f} [m/s]",
            f"RELATIVE_POSITION_R   = {float(self.relative_metadata['RELATIVE_POSITION_R']):.3f} [m]",
            f"RELATIVE_POSITION_T   = {float(self.relative_metadata['RELATIVE_POSITION_T']):.3f} [m]",
            f"RELATIVE_POSITION_N   = {float(self.relative_metadata['RELATIVE_POSITION_N']):.3f} [m]",
            f"COLLISION_PROBABILITY = {float(self.relative_metadata['COLLISION_PROBABILITY']):.8e}",
            f"COLLISION_PROB_METHOD = {self.relative_metadata['COLLISION_PROB_METHOD']}",
            "",
            "COMMENT Object 1 Details",
            f"OBJECT               = {self.object1['OBJECT']}",
            f"OBJECT_DESIGNATOR    = {self.object1['OBJECT_DESIGNATOR']}",
            f"CATALOG_NAME         = {self.object1['CATALOG_NAME']}",
            f"REF_FRAME            = {self.object1['REF_FRAME']}",
            "",
            "COMMENT Object 2 Details",
            f"OBJECT               = {self.object2['OBJECT']}",
            f"OBJECT_DESIGNATOR    = {self.object2['OBJECT_DESIGNATOR']}",
            f"CATALOG_NAME         = {self.object2['CATALOG_NAME']}",
            f"REF_FRAME            = {self.object2['REF_FRAME']}",
        ]
        return "\n".join(lines)

    def to_file(self, filepath: str | Path) -> None:
        """Save the CDM to a KVN file."""
        Path(filepath).write_text(self.to_string(), encoding="utf-8")

    def write(self, filepath: str | Path) -> None:
        """Alias for to_file."""
        self.to_file(filepath)

    @classmethod
    def from_file(cls, filepath: str | Path) -> CDM:
        """Parse a CDM from a KVN file."""
        cdm = cls()
        current_obj: dict[str, Any] | None = None
        for line in Path(filepath).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("COMMENT") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            clean_val = val.split("#")[0].split("[")[0].strip()

            if key == "OBJECT":
                current_obj = cdm.object1 if clean_val == "OBJECT1" else cdm.object2
                current_obj["OBJECT"] = clean_val
                continue

            if current_obj is not None and key in current_obj:
                try:
                    current_obj[key] = float(clean_val)
                except ValueError:
                    current_obj[key] = clean_val
            elif key in cdm.header:
                cdm.header[key] = clean_val
            elif key in cdm.relative_metadata:
                try:
                    cdm.relative_metadata[key] = float(clean_val)
                except ValueError:
                    cdm.relative_metadata[key] = clean_val
        return cdm
