import numpy as np
import pytest
from opengnc.utils.equinoctial_utils import kepler2equinoctial, equinoctial2eci
from opengnc.utils.mee_utils import kepler2mee, mee2eci
from opengnc.utils.state_to_elements import eci2kepler


def test_equinoctial_round_trip():
    a = 7000e3
    ecc = 0.01
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(60)
    M = np.radians(0)

    _, h, k, p, q, lm = kepler2equinoctial(a, ecc, incl, raan, argp, M)
    reci, veci = equinoctial2eci(a, h, k, p, q, lm)

    a_f, ecc_f, incl_f, raan_f, argp_f, nu_f, M_f, _, _, _, _, _ = eci2kepler(reci, veci)

    assert a == pytest.approx(a_f, rel=1e-5)
    assert ecc == pytest.approx(ecc_f, abs=1e-6)
    assert incl == pytest.approx(incl_f, abs=1e-6)


def test_mee_round_trip():
    a = 7000e3
    ecc = 0.01
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(60)
    nu = np.radians(0)

    p, f, g, h, k, L = kepler2mee(a, ecc, incl, raan, argp, nu)
    reci, veci = mee2eci(p, f, g, h, k, L)

    a_f, ecc_f, incl_f, raan_f, argp_f, nu_f, _, _, _, _, _, _ = eci2kepler(reci, veci)

    assert a == pytest.approx(a_f, rel=1e-5)
    assert ecc == pytest.approx(ecc_f, abs=1e-6)
    assert incl == pytest.approx(incl_f, abs=1e-6)


def test_equinoctial2kepler():
    from opengnc.utils.equinoctial_utils import equinoctial2kepler

    a_in = 7000e3
    h = 0.01
    k = 0.01
    p = 0.1
    q = 0.1
    lm = 1.0

    a, ecc, incl, raan, argp, nu, M = equinoctial2kepler(a_in, h, k, p, q, lm)
    assert a == a_in
    assert ecc > 0


def test_eci2mee():
    from opengnc.utils.mee_utils import eci2mee

    reci = np.array([7000e3, 0.0, 0.0])
    veci = np.array([0.0, 7500.0, 0.0])

    p, f, g, h, k, L = eci2mee(reci, veci)
    assert p > 0


def test_anomalies_singularities():
    from opengnc.utils.state_to_elements import anomalies

    E_h, M_h = anomalies(1.5, np.radians(30))
    assert np.isfinite(E_h) and np.isfinite(M_h)

    E_p, M_p = anomalies(1.0, np.radians(30))
    assert np.isfinite(E_p) and np.isfinite(M_p)


def test_state_to_elements_edge_cases():
    reci = np.array([7000e3, 0, 0])
    veci = np.array([0, -5000, 5000])  # retrograde/inclined
    elements = eci2kepler(reci, veci)
    assert np.isfinite(elements[0])


def test_circular_equatorial_edge_cases():
    from opengnc.utils.state_to_elements import eci2kepler

    mu = 398600.4415e9
    r_mag = 7000e3
    v_mag = np.sqrt(mu / r_mag)

    reci1 = np.array([0.0, -r_mag, 0.0])
    veci1 = np.array([v_mag, 0.0, 0.0])

    elements1 = eci2kepler(reci1, veci1)
    assert elements1[1] < 1e-5  # ecc
    assert elements1[2] < 1e-5  # incl
    assert np.isfinite(elements1[10])  # truelon

    reci2 = np.array([r_mag, 0.0, 0.0])
    veci2 = np.array([0.0, -v_mag, 0.0])

    elements2 = eci2kepler(reci2, veci2)
    assert elements2[1] < 1e-5  # ecc
    assert np.isclose(elements2[2], np.pi)  # incl ~ 180 deg
