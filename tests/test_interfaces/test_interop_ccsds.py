"""
Tests for CCSDS interfaces (OEM, OPM).
"""

import os
import tempfile

import numpy as np
import pytest

from opengnc.interfaces.ccsds import OEM, OPM


def test_opm_write_read():
    state = {
        "EPOCH": "2026-04-18T12:00:00.000",
        "X": 6678.14,
        "Y": 0.0,
        "Z": 0.0,
        "X_DOT": 0.0,
        "Y_DOT": 7.5,
        "Z_DOT": 0.0,
    }
    opm = OPM(state)

    with tempfile.NamedTemporaryFile(suffix=".opm", delete=False) as tmp:
        opm.write(tmp.name)
        tmp_name = tmp.name

    try:
        opm_loaded = OPM.from_file(tmp_name)
        assert opm_loaded.state["EPOCH"] == state["EPOCH"]
        assert float(opm_loaded.state["X"]) == state["X"]
        assert np.allclose(opm_loaded.get_state_vector()[:3], np.array([6678140.0, 0.0, 0.0]))
    finally:
        os.remove(tmp_name)


def test_oem_write_read():
    epochs = ["2026-04-18T12:00:00", "2026-04-18T12:10:00"]
    states = np.array([[6678.14, 0.0, 0.0, 0.0, 7.5, 0.0], [6500.0, 1000.0, 0.0, -0.5, 7.4, 0.0]])

    oem = OEM()
    oem.set_data(epochs, states)

    with tempfile.NamedTemporaryFile(suffix=".oem", delete=False) as tmp:
        oem.write(tmp.name)
        tmp_name = tmp.name

    try:
        oem_loaded = OEM.from_file(tmp_name)
        assert len(oem_loaded.data) == 2
        assert oem_loaded.metadata["START_TIME"] == epochs[0]
        assert np.allclose(oem_loaded.data.iloc[0].values, states[0])
    finally:
        os.remove(tmp_name)
