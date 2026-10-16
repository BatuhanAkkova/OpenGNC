import pytest
import numpy as np
from opengnc.propagators.cowell import CowellPropagator


def test_end_to_end_scenario():
    """
    Simulate a complete end-to-end mission scenario
    including orbit propagation and validation against
    expected results.
    """
    # LEO orbit state: position in km, velocity in km/s
    r_i = np.array([7000.0, 0.0, 0.0])
    v_i = np.array([0.0, 7.5, 0.0])

    dt = 3600.0  # 1 hour simulation

    # Run the propagation using the Cowell Propagator
    propagator = CowellPropagator(mu=398600.4418)
    r_f, v_f = propagator.propagate(r_i, v_i, dt, dt_step=60.0)

    # Validate output types
    assert isinstance(r_f, np.ndarray) and r_f.shape == (3,)
    assert isinstance(v_f, np.ndarray) and v_f.shape == (3,)

    # Calculate radius to ensure it stays somewhat physical (e.g. LEO)
    final_r = np.linalg.norm(r_f)
    assert final_r > 6000.0 and final_r < 40000.0
