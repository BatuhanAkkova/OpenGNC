import numpy as np
import pytest
from opengnc.utils.covariance import CovarianceTransform, ReachabilityAnalysis


def test_covariance_transform_eci_ric():
    """Verify ECI to RIC transformation logic."""
    r_eci = np.array([7000e3, 0, 0])
    v_eci = np.array([0, 7500.0, 0])

    # 6x6 Identity covariance
    P_eci = np.eye(6)

    P_ric = CovarianceTransform.eci_to_ric(r_eci, v_eci, P_eci)

    # Rotation of Identity should be Identity
    assert np.allclose(P_ric, np.eye(6))

    # Test reversibility
    P_back = CovarianceTransform.ric_to_eci(r_eci, v_eci, P_ric)
    assert np.allclose(P_back, P_eci)


def test_reachability_analysis():
    """Verify reachability budget calculation."""
    ra = ReachabilityAnalysis()
    a, e, i = 7000e3, 0.001, np.radians(28.5)
    dv = 10.0  # 10 m/s

    res = ra.get_reachable_delta_elements(a, e, i, dv)

    assert res["max_delta_a"] > 0
    assert res["max_delta_e"] > 0
    assert res["max_delta_i"] > 0
    print(res)


if __name__ == "__main__":
    test_covariance_transform_eci_ric()
    test_reachability_analysis()
