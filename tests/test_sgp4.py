import pytest
import numpy as np
from opengnc.propagators.sgp4_propagator import Sgp4Propagator

pytest.importorskip("sgp4")

def test_sgp4_init_and_propagate():
    # ISS TLE (Sample)
    line1 = "1 25544U 98067A   20325.43545139  .00001564  00000-0  36550-4 0  9997"
    line2 = "2 25544  51.6441 332.1863 0001461 319.4678  26.1557 15.49163286256157"
    
    sgp4_prop = Sgp4Propagator(line1, line2)
    
    # Propagate 1 orbit forward
    dt = 5400.0
    r_f, v_f = sgp4_prop.propagate(None, None, dt)
    
    assert len(r_f) == 3
    assert len(v_f) == 3
    
    # Check altitude magnitude is reasonable for LEO
    r_mag = np.linalg.norm(r_f)
    print(f"SGP4 R_mag: {r_mag/1000} km")
    assert 6700e3 < r_mag < 6900e3
    
    v_mag = np.linalg.norm(v_f)
    print(f"SGP4 V_mag: {v_mag/1000} km/s")
    assert 7000 < v_mag < 8000

def test_sgp4_propagate_to_jd():
    line1 = "1 25544U 98067A   20325.43545139  .00001564  00000-0  36550-4 0  9997"
    line2 = "2 25544  51.6441 332.1863 0001461 319.4678  26.1557 15.49163286256157"
    sgp4_prop = Sgp4Propagator(line1, line2)
    
    # Propagate to arbitrary JD
    jd_f = sgp4_prop.jdsatepoch + 0.1 # 0.1 days forward
    r_f, v_f = sgp4_prop.propagate_to_jd(jd_f, 0.0)
    
    assert len(r_f) == 3
    assert len(v_f) == 3

def test_sgp4_errors():
    from unittest.mock import patch, MagicMock
    from opengnc.propagators.sgp4_propagator import Sgp4Propagator
    import pytest
    
    line1 = "1 25544U 98067A   20325.43545139  .00001564  00000-0  36550-4 0  9997"
    line2 = "2 25544  51.6441 332.1863 0001461 319.4678  26.1557 15.49163286256157"
    
    mock_sat = MagicMock()
    mock_sat.jdsatepoch = 2440000.0
    mock_sat.jdsatepochF = 0.5
    mock_sat.sgp4.return_value = (1, [0,0,0], [0,0,0])
    
    with patch('opengnc.propagators.sgp4_propagator.Satrec.twoline2rv', return_value=mock_sat):
        prop = Sgp4Propagator(line1, line2)
        
        with pytest.raises(RuntimeError):
            prop.propagate(None, None, 100.0)
            
        with pytest.raises(RuntimeError):
            prop.propagate_to_jd(2451545.0)




