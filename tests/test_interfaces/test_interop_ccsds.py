"""
Tests for CCSDS interfaces (OEM, OPM, OMM, CDM).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from opengnc.interfaces.ccsds import CDM, OEM, OMM, OPM


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


def test_omm_write_read_and_tle():
    line1 = "1 25544U 98067A   24100.52840278  .00016717  00000+0  10270-3 0  9022"
    line2 = "2 25544  51.6416 250.3120 0004561  87.2100 272.9300 15.49815012448324"

    omm = OMM.from_tle(line1, line2, name="ISS (ZARYA)")
    assert omm.metadata["OBJECT_NAME"] == "ISS (ZARYA)"
    assert float(omm.mean_elements["INCLINATION"]) == pytest.approx(51.6416)
    assert float(omm.mean_elements["MEAN_MOTION"]) == pytest.approx(15.49815012)
    assert float(omm.mean_elements["ECCENTRICITY"]) == pytest.approx(0.0004561)

    with tempfile.NamedTemporaryFile(suffix=".omm", delete=False) as tmp:
        omm.write(tmp.name)
        tmp_name = tmp.name

    try:
        omm_loaded = OMM.from_file(tmp_name)
        assert omm_loaded.metadata["OBJECT_NAME"] == "ISS (ZARYA)"
        assert float(omm_loaded.mean_elements["INCLINATION"]) == pytest.approx(51.6416)
        elems = omm_loaded.get_keplerian_elements()
        assert elems["raan"] == pytest.approx(250.3120)
    finally:
        os.remove(tmp_name)


def test_cdm_write_read_and_collision_probability():
    cdm = CDM()
    cdm.relative_metadata["TCA"] = "2026-05-01T04:22:15.120"
    cdm.relative_metadata["MISS_DISTANCE"] = 45.2
    cdm.relative_metadata["RELATIVE_POSITION_R"] = 35.0
    cdm.relative_metadata["RELATIVE_POSITION_N"] = 28.6

    pc = cdm.compute_collision_probability(hard_body_radius=5.0)
    assert 0.0 <= pc <= 1.0
    assert pc > 0.0

    with tempfile.NamedTemporaryFile(suffix=".cdm", delete=False) as tmp:
        cdm.write(tmp.name)
        tmp_name = tmp.name

    try:
        cdm_loaded = CDM.from_file(tmp_name)
        assert cdm_loaded.relative_metadata["TCA"] == "2026-05-01T04:22:15.120"
        assert float(cdm_loaded.relative_metadata["MISS_DISTANCE"]) == pytest.approx(45.2)
        assert float(cdm_loaded.relative_metadata["COLLISION_PROBABILITY"]) == pytest.approx(pc)
    finally:
        os.remove(tmp_name)
