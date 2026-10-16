import pytest
import numpy as np
from opengnc.mission_design.budgeting import (
    calculate_propellant_mass,
    calculate_delta_v,
    calculate_staged_delta_v,
    ManeuverSequence,
    predict_lifetime,
)
from opengnc.environment.density import Exponential


def test_calculate_propellant_mass():
    m0 = 1000.0  # kg
    dv = 0.5  # km/s
    isp = 300.0  # s
    g0 = 0.00980665  # km/s^2

    m_prop = calculate_propellant_mass(m0, dv, isp)

    # Analytical: m_f = m0 / exp(dv / (isp * g0))
    # m_prop = m0 - m_f
    expected_mf = m0 / np.exp(dv / (isp * g0))
    expected_m_prop = m0 - expected_mf

    assert m_prop == pytest.approx(expected_m_prop)


def test_calculate_delta_v():
    m0 = 1000.0
    m_prop = 150.0
    isp = 300.0
    g0 = 0.00980665

    dv = calculate_delta_v(m0, m_prop, isp)

    expected_dv = isp * g0 * np.log(m0 / (m0 - m_prop))
    assert dv == pytest.approx(expected_dv)


def test_calculate_staged_delta_v():
    # 2 stage rocket
    stages = [
        {"m_dry": 100, "m_prop": 500, "isp": 300},  # Stage 1 (Bottom)
        {"m_dry": 50, "m_prop": 200, "isp": 320},  # Stage 2 (Top)
    ]

    # Manual math:
    # Top stage:
    # m0_2 = 50 + 200 = 250 kg
    # mf_2 = 50 kg
    # dv2 = 320 * 0.00980665 * ln(250/50) = 320 * g0 * ln(5)

    # Bottom stage:
    # m0_1 = 100 + 500 + 250 = 850 kg
    # mf_1 = 100 + 250 = 350 kg
    # dv1 = 300 * 0.00980665 * ln(850/350)

    g0 = 0.00980665
    dv2 = 320 * g0 * np.log(250.0 / 50.0)
    dv1 = 300 * g0 * np.log(850.0 / 350.0)
    expected_total_dv = dv1 + dv2

    total_dv = calculate_staged_delta_v(stages)
    assert total_dv == pytest.approx(expected_total_dv)


def test_maneuver_sequence():
    m0 = 1000.0
    isp = 300.0
    seq = ManeuverSequence(m0, isp)

    seq.add_maneuver("Burn 1", 0.2)
    seq.add_maneuver("Burn 2", 0.3)

    history = seq.get_budget_history()
    assert len(history) == 2
    assert history[0]["name"] == "Burn 1"
    assert history[1]["name"] == "Burn 2"

    # Check total propellant
    # seq.current_mass should match m0 - total_propellant
    m_f_manual = m0
    for dv in [0.2, 0.3]:
        m_prop = calculate_propellant_mass(m_f_manual, dv, isp)
        m_f_manual -= m_prop

    assert seq.current_mass == pytest.approx(m_f_manual)
    assert seq.get_total_propellant() == pytest.approx(m0 - m_f_manual)
    assert seq.get_total_dv() == 0.5


def test_predict_lifetime_fast_decay():
    # Create a scenario with VERY high drag or very low orbit for immediate decay
    # Reentry at 100km altitude
    # Place orbit at 105km altitude

    # Position: 105km altitude above Earth (6378.137 km)
    # r = [6483.137, 0, 0] km
    r_eci = np.array([6483.137, 0, 0])
    v_eci = np.array([0, 7.8, 0])  # rough circular velocity

    mass = 100.0  # kg
    area = 10.0  # m^2 (Aggressive area)
    cd = 2.2

    # Dense density model to speed up decay
    density_model = Exponential(rho0=1e-6, h0=100.0, H=5.0)  # extreme density
    jd_epoch = 2460000.5

    res = predict_lifetime(
        r_eci, v_eci, mass, area, cd, density_model, jd_epoch, max_days=1, dt=1.0
    )

    assert res["reentry_detected"] is True
    assert res["lifetime_days"] > 0
    assert res["final_altitude"] == pytest.approx(
        100000.0, abs=5000
    )  # within tolerance 100km altitude


def test_budgeting_exceptions():
    from opengnc.mission_design.budgeting import (
        calculate_propellant_mass,
        calculate_delta_v,
        ManeuverSequence,
    )
    import pytest

    with pytest.raises(ValueError, match="Isp must be positive"):
        calculate_propellant_mass(1000, 1.0, 0)
    with pytest.raises(ValueError, match="Isp must be positive"):
        calculate_delta_v(1000, 100, 0)
    with pytest.raises(ValueError, match="Propellant mass cannot exceed"):
        calculate_delta_v(1000, 1500, 300)

    seq = ManeuverSequence(1000, 300)
    with pytest.raises(ValueError, match="Delta-V must be non-negative"):
        seq.add_maneuver("test", -1.0)
