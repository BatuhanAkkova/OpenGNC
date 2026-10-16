import numpy as np
import pytest
from opengnc.mission_design import (
    calculate_access_windows,
    calculate_ground_track,
    calculate_lighting_conditions,
    calculate_launch_windows,
    compute_injection_state,
    calculate_constellation_coverage,
    calculate_deployment_sequence,
    calculate_doppler_shift,
    calculate_atmospheric_attenuation,
    calculate_friis_link_budget,
)
from opengnc.utils.frame_conversion import llh2ecef, ecef2eci
from unittest.mock import patch
from opengnc.environment.solar import Sun


def test_calculate_access_windows():
    t_array = np.array([0, 100, 200])

    gs_lat = 0.0
    gs_lon = 0.0
    gs_alt = 0.0

    R_earth = 6378137.0

    r_gs_ecef = llh2ecef(0, 0, 0)  # [R_earth, 0, 0]
    r_sat_ecef_above = r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat_eci_above, _ = ecef2eci(r_sat_ecef_above, np.zeros(3), 2451545.0)

    r_sat_ecef_below = -r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat_eci_below, _ = ecef2eci(r_sat_ecef_below, np.zeros(3), 2451545.0)

    r_eci_array = np.vstack((r_sat_eci_above, r_sat_eci_below, r_sat_eci_above))

    res = calculate_access_windows(
        t_array, r_eci_array, gs_lat, gs_lon, gs_alt, min_elevation_deg=5.0, jdut1=2451545.0
    )

    assert "visible_intervals" in res
    assert "elevation_history" in res
    assert len(res["visible_intervals"]) == 2  # 0 and 200 visible
    assert res["elevation_history"][0] > 80.0  # Directly above
    assert res["elevation_history"][1] < 0.0  # Below horizon


def test_calculate_ground_track():
    t_array = np.array([0])
    R_earth = 6378137.0
    r_eci = np.array([R_earth + 500000.0, 0, 0])

    res = calculate_ground_track(t_array, np.array([r_eci]), jdut1=2451545.0)

    assert "lat_deg" in res
    assert "lon_deg" in res
    assert "alt_m" in res
    assert len(res["lat_deg"]) == 1


def test_calculate_lighting_conditions():
    t_array = np.array([0])
    R_earth = 6378137.0
    r_eci = np.array([0, R_earth + 500000.0, 0])
    v_eci = np.array([1, 0, 0])  # Just some velocity

    res = calculate_lighting_conditions(
        t_array, np.array([r_eci]), np.array([v_eci]), jdut1=2451545.0
    )

    assert "beta_angle_deg" in res
    assert "eclipse_state" in res
    assert len(res["beta_angle_deg"]) == 1


def test_calculate_launch_windows():
    jd_start = 2451545.0
    jd_end = 2451546.0  # 1 day
    inc = 45.0
    raan = 0.0
    lat = 28.5  # Cape Canaveral
    lon = -80.5

    res = calculate_launch_windows(jd_start, jd_end, inc, raan, lat, lon)

    assert isinstance(res, list)
    assert len(res) >= 1


def test_compute_injection_state():
    lat = 28.5
    lon = -80.5
    alt = 200000.0
    azimuth = 90.0  # Eastward
    fpa = 0.0  # Horizontal
    speed = 7800.0  # Orbital speed m/s
    jd = 2451545.0

    r_eci, v_eci = compute_injection_state(lat, lon, alt, azimuth, fpa, speed, jd)

    assert len(r_eci) == 3
    assert len(v_eci) == 3
    assert abs(np.linalg.norm(r_eci) - (6378137.0 + alt)) < 20000.0


def test_calculate_constellation_coverage():
    t_array = np.array([0, 100, 200])

    target_points = np.array([[0.0, 0.0, 0.0]])

    R_earth = 6378137.0

    r_gs_ecef = llh2ecef(0, 0, 0)
    r_sat1_ecef = r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat1_eci, _ = ecef2eci(r_sat1_ecef, np.zeros(3), 2451545.0)

    r_sat2_ecef = -r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat2_eci, _ = ecef2eci(r_sat2_ecef, np.zeros(3), 2451545.0)

    r_eci_array_1 = np.vstack((r_sat1_eci, r_sat2_eci, r_sat1_eci))
    r_eci_array_2 = np.vstack((r_sat2_eci, r_sat1_eci, r_sat2_eci))
    r_eci_array_list = [r_eci_array_1, r_eci_array_2]

    res = calculate_constellation_coverage(
        t_array, r_eci_array_list, target_points, min_elevation_deg=5.0, jdut1=2451545.0
    )

    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["max_revisit_time_gap"] == 0.0
    assert res[0]["total_coverage_time"] > 0


def test_calculate_deployment_sequence():
    planes = 3
    sats_per_plane = 2
    phasing_f = 1

    seq = calculate_deployment_sequence(planes, sats_per_plane, phasing_f, inc_deg=45.0)

    assert len(seq) == 6

    assert seq[0]["raan_deg"] == 0.0
    assert seq[0]["ta_deg"] == 0.0

    assert seq[1]["raan_deg"] == 0.0
    assert seq[1]["ta_deg"] == 180.0

    assert seq[2]["raan_deg"] == 120.0
    assert seq[2]["ta_deg"] == 60.0


def test_communications_doppler_distance_zero():
    r = np.array([1000, 0, 0])
    v = np.array([0, 1, 0])
    res = calculate_doppler_shift(1e9, r, v, r, v)
    assert res["doppler_shift_hz"] == 0.0

    att1 = calculate_atmospheric_attenuation(30, 5e9)
    assert att1 > 0
    att2 = calculate_atmospheric_attenuation(30, 15e9)
    assert att2 > 0


def test_coverage_other_branches():
    r = np.array([[7000e3, 0, 0]])
    v = np.array([[7e3, 0, 0]])
    res = calculate_lighting_conditions(np.array([0]), r, v)
    assert res["beta_angle_deg"][0] == 0.0

    with patch("opengnc.mission_design.coverage.eci2llh", return_value=(0, 4.0, 0)):
        res = calculate_ground_track(np.array([0]), np.array([[1, 0, 0]]))
        assert res["lon_deg"][0] < 180.0

    with patch("opengnc.mission_design.coverage.eci2llh", return_value=(0, -4.0, 0)):
        res = calculate_ground_track(np.array([0]), np.array([[1, 0, 0]]))
        assert res["lon_deg"][0] > -180.0

    t_array = np.array([0, 10, 20, 30])
    sats = [np.array([[7000e3, 0, 0], [7000e3, 0, 0], [7000e3, 0, 0], [7000e3, 0, 0]])]
    targets = np.array([[0, 0]])
    with patch(
        "opengnc.mission_design.coverage.eci2ecef", return_value=(np.array([1, 1, 1]), None)
    ):
        calculate_constellation_coverage(t_array, sats, targets)


def test_launch_azimuth_edge():
    jd = 2451545.0
    with patch("opengnc.mission_design.launch.eci2_ecef_or_inverse_wrapper") as mock_wrapper:
        mock_wrapper.side_effect = lambda r, jd: (
            np.array([0, 1e6 if int(jd * 1e5) % 2 == 0 else -1e6, 0]),
            None,
        )

        res = calculate_launch_windows(jd, jd + 0.1, 10.0, 0, 45.0, 0)
        assert res is not None


def test_launch_pole_edge():
    jd = 2451545.0
    with patch("opengnc.mission_design.launch.eci2_ecef_or_inverse_wrapper") as mock_wrapper:
        mock_wrapper.side_effect = lambda r, jd: (
            np.array([0, 1e6 if int(jd * 10) % 2 == 0 else -1e6, 0]),
            None,
        )
        res = calculate_launch_windows(jd, jd + 0.1, 45.0, 0, 90.0, 0)
        assert res is not None


def test_calculate_access_windows_rho_zero():
    t_array = np.array([0])
    r_eci_array = np.array([[7000e3, 0, 0]])
    gs_lat = 0.0
    gs_lon = 0.0
    gs_alt = 0.0

    r_gs_ecef = llh2ecef(0, 0, 0)

    with patch("opengnc.mission_design.coverage.eci2ecef", return_value=(r_gs_ecef, None)):
        res = calculate_access_windows(t_array, r_eci_array, gs_lat, gs_lon, gs_alt)
        assert res["elevation_history"][0] == 90.0
        assert res["visible_intervals"][0]["start_time"] == 0


def test_calculate_lighting_conditions_outside_shadow():
    t_array = np.array([0])
    R_earth = 6378137.0
    r_eci = np.array([-R_earth, R_earth + 1000.0, 0.0])
    v_eci = np.array([0.0, 0.0, 0.0])

    with patch.object(Sun, "calculate_sun_eci", return_value=np.array([1.0, 0.0, 0.0])):
        res = calculate_lighting_conditions(t_array, np.array([r_eci]), np.array([v_eci]))
        assert res["eclipse_state"][0] == 1.0  # Sunlight


def test_calculate_constellation_coverage_gap_transitions():
    t_array = np.array([0, 10, 20, 30])
    r_eci = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    r_eci_list = [r_eci]
    targets = np.array([[0.0, 0.0, 0.0]])

    r_gs_ecef = llh2ecef(0, 0, 0)  # [R_earth, 0, 0]

    r_ecef_0 = r_gs_ecef + np.array([100.0, 0, 0])
    r_ecef_1 = r_gs_ecef + np.array([-100.0, 0, 0])

    mock_returns = [
        (r_ecef_0, None),
        (r_ecef_1, None),
        (r_ecef_1, None),
        (r_ecef_0, None),
    ]

    with patch("opengnc.mission_design.coverage.eci2ecef", side_effect=mock_returns):
        res = calculate_constellation_coverage(t_array, r_eci_list, targets, min_elevation_deg=0.0)
        assert len(res) == 1
        assert res[0]["total_coverage_time"] > 0
        assert res[0]["max_revisit_time_gap"] > 0


def test_calculate_constellation_coverage_rho_zero():
    t_array = np.array([0])
    r_eci_list = [np.array([[7000e3, 0, 0]])]
    targets = np.array([[0.0, 0.0, 0.0]])

    r_gs_ecef = llh2ecef(0, 0, 0)

    with patch("opengnc.mission_design.coverage.eci2ecef", return_value=(r_gs_ecef, None)):
        res = calculate_constellation_coverage(t_array, r_eci_list, targets)
        assert len(res) == 1
        assert res[0]["total_coverage_time"] == 0.0  # Just one point, dt=0


# --- Communications Tests ---


def test_friis_link_budget():
    p_tx_w = 10.0  # 10 W -> 10 dBW
    g_tx_db = 10.0
    g_rx_db = 15.0
    frequency_hz = 2.0e9  # 2 GHz (S-band)
    distance_m = 1000.0e3  # 1000 km

    # Lfs = 20*log10(1000e3) + 20*log10(2.0e9) - 147.554
    # = 120 + 186.02 - 147.554 = 158.466

    result = calculate_friis_link_budget(
        p_tx_w=p_tx_w,
        g_tx_db=g_tx_db,
        g_rx_db=g_rx_db,
        frequency_hz=frequency_hz,
        distance_m=distance_m,
    )

    assert "p_rx_dbw" in result
    assert "p_rx_w" in result
    assert "l_fs_db" in result

    # Check expected values
    # Lfs should be approx 158.466
    assert result["l_fs_db"] == pytest.approx(158.466, abs=0.1)

    # Prx (dBW) = 10 + 10 + 15 - 158.466 = -123.466
    assert result["p_rx_dbw"] == pytest.approx(-123.466, abs=0.1)

    # Include losses
    result_losses = calculate_friis_link_budget(
        p_tx_w=p_tx_w,
        g_tx_db=g_tx_db,
        g_rx_db=g_rx_db,
        frequency_hz=frequency_hz,
        distance_m=distance_m,
        losses_misc_db=2.0,
        l_atm_db=1.0,
    )

    # Prx (dBW) = -123.466 - 2 - 1 = -126.466
    assert result_losses["p_rx_dbw"] == pytest.approx(-126.466, abs=0.1)


def test_doppler_shift():
    f_tx_hz = 2.0e9  # 2 GHz

    # Stationary Tx at origin
    r_tx = np.array([0.0, 0.0, 0.0])
    v_tx = np.array([0.0, 0.0, 0.0])

    # Rx moving away along X axis at 100 m/s
    r_rx = np.array([1000.0, 0.0, 0.0])
    v_rx = np.array([100.0, 0.0, 0.0])

    result = calculate_doppler_shift(f_tx_hz, r_rx, v_rx, r_tx, v_tx)

    # Expected shift: - f_tx * (v_rel / c)
    # v_rel = 100 m/s
    # c = 299792458
    # shift = - 2e9 * 100 / 299792458 = -667.128 Hz

    assert result["doppler_shift_hz"] == pytest.approx(-667.128, abs=0.001)
    assert result["f_rx_hz"] == pytest.approx(f_tx_hz - 667.128, abs=0.001)

    # Rx moving closer
    v_rx_closer = np.array([-100.0, 0.0, 0.0])
    result_closer = calculate_doppler_shift(f_tx_hz, r_rx, v_rx_closer, r_tx, v_tx)
    assert result_closer["doppler_shift_hz"] == pytest.approx(667.128, abs=0.001)


def test_atmospheric_attenuation():
    # S-band, Elevation 30 deg
    # Latm = 0.03 / sin(30) = 0.03 / 0.5 = 0.06 dB
    elevation = 30.0
    freq_s = 2.0e9

    l_atm_s = calculate_atmospheric_attenuation(elevation, freq_s)
    assert l_atm_s == pytest.approx(0.06)

    # Ka-band, Elevation 45 deg
    # Azentih = 0.35
    # Latm = 0.35 / sin(45) = 0.35 / 0.707106 = 0.49497 dB
    freq_ka = 20.0e9
    l_atm_ka = calculate_atmospheric_attenuation(45.0, freq_ka)
    assert l_atm_ka == pytest.approx(0.49497, abs=0.001)

    # Low elevation cap (should cap at 5 deg)
    l_atm_low = calculate_atmospheric_attenuation(1.0, freq_s)
    l_atm_cap = calculate_atmospheric_attenuation(5.0, freq_s)
    assert l_atm_low == l_atm_cap


def test_communications_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_friis_link_budget(-1, 10, 10, 1e9, 1000)
    with pytest.raises(ValueError):
        calculate_friis_link_budget(10, 10, 10, -1, 1000)
    with pytest.raises(ValueError):
        calculate_friis_link_budget(10, 10, 10, 1e9, -1)

    # Doppler shape error
    with pytest.raises(ValueError):
        calculate_doppler_shift(
            1e9, np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])
        )
