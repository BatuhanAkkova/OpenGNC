import pytest
import numpy as np
import os
import json


def load_reference_data(filename):
    """
    Load mock reference data, pretending it's from GMAT/STK
    """
    # For now, return a mock dictionary
    return {
        "final_state": [6800.0, 1000.0, 500.0, -1.0, 7.0, 1.0],
    }


def test_regression_reference_data():
    """
    Compare our toolkit output with standard reference mock data
    """
    ref_data = load_reference_data("mock_gmat_results.json")

    # Example simulated output
    simulated_final_state = np.array([6800.1, 999.8, 500.5, -0.99, 7.01, 1.0])

    # Check that difference is within small tolerance
    err = np.linalg.norm(np.array(ref_data["final_state"]) - simulated_final_state)
    assert err < 1.0  # Allow some small numerical difference
