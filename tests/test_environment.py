import pytest
import numpy as np
import datetime
from opengnc.environment.density import Exponential, HarrisPriester, NRLMSISE00, JB2008, CIRA72
from opengnc.environment.mag_field import tilted_dipole_field, igrf_field, wmm_field
from opengnc.environment.space_weather import SpaceWeather
from opengnc.environment.radiation import RadiationModel
from opengnc.environment.thermal import ThermalEnvironment
from opengnc.environment.wind import AtmosphereCoRotation
from opengnc.environment.moon import Moon
from opengnc.utils.time_utils import calc_jd
import sys
import importlib


def test_exponential_model():
    model = Exponential()
    r_eci = np.array([6500e3, 0.0, 0.0])
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0.0
    assert rho < 1.225

    r_space = np.array([50000e3, 0.0, 0.0])
    rho_space = model.get_density(r_space, jd)
    assert rho_space < rho
    assert rho_space >= 0.0


def test_harris_priester_model():
    model = HarrisPriester()
    r_eci = np.array([7000e3, 0.0, 0.0])
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0.0
    assert rho < 1.225

    r_space = np.array([50000e3, 0.0, 0.0])
    rho_space = model.get_density(r_space, jd)
    assert rho_space < rho
    assert rho_space >= 0.0


def test_nrlmsise_model(mocker):
    # Mock pymsis.calculate
    mock_calc = mocker.patch("pymsis.calculate")
    mock_calc.return_value = np.array([1.5e-12])  # Mock density

    model = NRLMSISE00()
    r_eci = np.array([7000e3, 0.0, 0.0])
    date = datetime.datetime(2024, 1, 1, 12, 0, 0)

    rho = model.get_density(r_eci, date)
    assert isinstance(rho, float)
    assert rho == 1.5e-12
    mock_calc.assert_called_once()


def test_tilted_dipole_field():
    # Test at some random position
    r_ecef = np.array([7000000.0, 0, 0])
    B = tilted_dipole_field(r_ecef)

    # Check shape and non-zero
    assert B.shape == (3,)
    assert np.linalg.norm(B) > 0.0

    # Check magnitude roughly (at 7000km, should be in microTesla range)
    # B approx: 3.12e-5 * (6371/7000)^3 = 2e-5 T
    assert np.linalg.norm(B) < 1e-4
    assert np.linalg.norm(B) > 1e-6


def test_igrf_field_mock(mocker):
    # Mock ppigrf.igrf via the module reference for robustness
    import opengnc.environment.mag_field as mag_field

    mock_igrf = mocker.patch.object(mag_field.ppigrf, "igrf")
    mock_igrf.return_value = [1e-5, 2e-5, 3e-5]

    lat = 0.0
    lon = 0.0
    alt = 600.0  # km
    date = datetime.datetime(2024, 1, 1)

    res = igrf_field(lat, lon, alt, date)
    assert len(res) == 3
    assert np.allclose(res, [1e-5, 2e-5, 3e-5])
    mock_igrf.assert_called_once()


def test_exponential_model_below_sea_level():
    model = Exponential()
    # R_earth is 6378.137 km in model. Position < R_earth.
    r_eci = np.array([6000e3, 0.0, 0.0])
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho == model.rho0


def test_density_h_scale_alias_and_low_alt():
    e = Exponential(h_scale=10.0)
    assert e.h_scale == 10.0
    hp = HarrisPriester()
    r = np.array([6371e3 + 50e3, 0, 0])
    rho = hp.get_density(r, 2460000.5)
    assert rho == 4.974e-07


def test_nrlmsise_model_array_output(mocker):
    mock_calc = mocker.patch("pymsis.calculate")
    # Return multiple elements so np.squeeze leaves ndim=1
    mock_calc.return_value = np.array([1.5e-12, 2.0e-12])

    model = NRLMSISE00()
    r_eci = np.array([7000e3, 0.0, 0.0])
    date = datetime.datetime(2024, 1, 1, 12, 0, 0)

    rho = model.get_density(r_eci, date)
    assert rho == 1.5e-12  # Should take output[0]


def test_cira72_model_low_altitude():
    from opengnc.environment.density import CIRA72

    model = CIRA72()
    # Altitude < 100 km (e.g., 50 km) -> h_km = 50
    # geodetic height h above Re
    r_eci = np.array([6378137.0 + 50000.0, 0, 0])  # approximate
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0


def test_igrf_field_import_error(mocker):
    import opengnc.environment.mag_field as mag_field

    mocker.patch.object(mag_field, "ppigrf", None)

    with pytest.raises(ImportError, match="ppigrf not installed"):
        mag_field.igrf_field(0, 0, 600, datetime.datetime(2024, 1, 1))


def test_mag_field_small_r():
    b = tilted_dipole_field(np.array([0.5, 0.0, 0.0]))
    assert np.allclose(b, 0.0)


def test_mag_field_module_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ppigrf", None)
    import opengnc.environment.mag_field

    importlib.reload(opengnc.environment.mag_field)
    assert opengnc.environment.mag_field.ppigrf is None
    del sys.modules["ppigrf"]
    importlib.reload(opengnc.environment.mag_field)


# --- Extended Environment Tests ---


def test_wmm_field_mock(mocker):
    # Mock ppigrf.igrf via the module reference
    import opengnc.environment.mag_field as mag_field

    mock_igrf = mocker.patch.object(mag_field.ppigrf, "igrf")
    mock_igrf.return_value = [1.1e-5, 2.1e-5, 3.1e-5]

    lat, lon, alt = 45.0, -120.0, 500.0
    date = datetime.datetime(2025, 1, 1)

    res = wmm_field(lat, lon, alt, date)
    assert len(res) == 3
    assert np.allclose(res, [1.1e-5, 2.1e-5, 3.1e-5])


def test_space_weather():
    sw = SpaceWeather(f107=140.0, ap=20.0)
    indices = sw.get_indices()
    assert indices["f107"] == 140.0
    assert indices["ap"] == 20.0
    assert indices["kp"] > 0

    sw.set_solar_flux(180.0)
    assert sw.f107 == 180.0
    assert sw.f107_avg == 180.0


def test_radiation_model():
    rad = RadiationModel()
    tid = rad.estimate_tid(500, 51.6, 365)
    assert tid > 0

    seu_rate = rad.estimate_seu_rate(500)
    assert seu_rate > 0


def test_thermal_environment():
    thermal = ThermalEnvironment(albedo_coeff=0.3)

    # Test Solar Flux
    solar = thermal.get_solar_flux(1.0)
    assert np.isclose(solar, 1361.0)

    # Test Albedo (Nadir facing Sat-Sun aligned)
    r_sat = np.array([7000, 0, 0])
    r_sun = np.array([149e6, 0, 0])
    albedo = thermal.get_albedo_flux(r_sat, r_sun)
    assert albedo > 0
    assert albedo < 1361.0 * 0.35  # Should be around 1361 * 0.3

    # Night side Albedo
    r_sun_night = np.array([-149e6, 0, 0])
    albedo_night = thermal.get_albedo_flux(r_sat, r_sun_night)
    assert albedo_night == 0.0

    # Earth IR
    ir = thermal.get_earth_ir_flux()
    assert ir == 230.0


def test_space_weather_set_solar_flux_avg():
    sw = SpaceWeather(f107=140.0, ap=20.0)
    sw.set_solar_flux(180.0, 170.0)
    assert sw.f107 == 180.0
    assert sw.f107_avg == 170.0


def test_space_weather_ap_zero():
    sw = SpaceWeather(ap=0.0)
    assert sw.kp == 0.0


def test_atmosphere_wind_relative_velocity():
    from opengnc.environment.wind import AtmosphereCoRotation

    model = AtmosphereCoRotation()
    r_eci = np.array([7000e3, 0.0, 0.0])
    v_eci = np.array([0.0, 7500.0, 0.0])
    jd = 2451545.0

    v_rel = model.get_relative_velocity(r_eci, v_eci, jd)
    assert v_rel.shape == (3,)
    # Wind with omega_z should affect Y-velocity for X-position
    assert not np.allclose(v_rel, v_eci)


# --- Environment Models Tests ---


def test_jb2008_density():
    model = JB2008()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0  # 622 km altitude
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0
    assert rho < 1e-6


def test_cira72_density():
    model = CIRA72()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    jd = 2451545.0
    rho1 = model.get_density(r_eci, jd)

    r_eci2 = np.array([7100.0, 0.0, 0.0]) * 1000.0
    rho2 = model.get_density(r_eci2, jd)

    assert rho1 > rho2  # Density should decrease with altitude
    assert rho1 > 0


def test_atmosphere_corotation():
    model = AtmosphereCoRotation()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0  # Equator
    jd = 2451545.0
    v_wind = model.get_wind_velocity(r_eci, jd)

    # At equator (7000km), v = omega * r
    # v ~ 7.29e-5 * 7e6 ~ 510 m/s
    assert np.allclose(v_wind, [0, 510.448, 0], atol=1.0)

    r_pole = np.array([0, 0, 7000.0]) * 1000.0
    v_wind_pole = model.get_wind_velocity(r_pole, jd)
    assert np.allclose(v_wind_pole, [0, 0, 0])


def test_moon_position():
    model = Moon()
    jd = 2451545.0  # J2000
    r_moon = model.calculate_moon_eci(jd)

    # Moon distance is approx 384,400 km
    dist = np.linalg.norm(r_moon) / 1000.0
    assert dist > 350000 and dist < 410000
