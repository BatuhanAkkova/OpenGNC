from __future__ import annotations

from pathlib import Path
from typing import Any

from .akf import AKF
from .attitude_fusion import AttitudeSensorFusion
from .ckf import CKF
from .ekf import EKF
from .enkf import EnKF
from .imm import IMM
from .kf import KF
from .mekf import MEKF as PythonMEKF
from .pf import ParticleFilter
from .rts_smoother import rts_smoother
from .sr_ukf import SRUKF
from .ukf import UKF
from .ukf import UKF_Attitude as PythonUKF_Attitude

MEKF: Any = PythonMEKF
UKF_Attitude: Any = PythonUKF_Attitude
ACCELERATION_AVAILABLE = False

try:
    import sys

    import opengnc_py

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent.parent
    build_paths = [
        project_root / "build" / "Release",
        project_root / "build",
        project_root / "cpp" / "build" / "Release",
        project_root / "cpp" / "build",
    ]

    for path in build_paths:
        if path.exists():
            sys.path.append(str(path))

    MEKF = opengnc_py.MEKF
    UKF_Attitude = opengnc_py.UKF_Attitude
    ACCELERATION_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "ACCELERATION_AVAILABLE",
    "AKF",
    "AttitudeSensorFusion",
    "CKF",
    "EKF",
    "IMM",
    "KF",
    "MEKF",
    "SRUKF",
    "UKF",
    "EnKF",
    "ParticleFilter",
    "PythonMEKF",
    "PythonUKF_Attitude",
    "RTS_smoother",
    "UKF_Attitude",
]

RTS_smoother = rts_smoother
