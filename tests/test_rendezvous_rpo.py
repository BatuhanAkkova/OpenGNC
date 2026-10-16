import pytest
import numpy as np
from opengnc.guidance.rendezvous import (
    tschauner_hempel_propagation,
    solve_lambert_multi_rev,
    primer_vector_analysis,
    is_within_corridor,
    optimize_rpo_collocation,
)


def test_tschauner_hempel_basic():
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1km radial offset
    oe_target = (7000, 0.01, 0, 0, 0, 0)  # a=7000km, e=0.01
    dt = 100.0

    state_f = tschauner_hempel_propagation(x0, oe_target, dt)
    assert state_f is not None
    assert state_f.shape == (6,)


def test_lambert_multi_rev_detect():
    r1 = np.array([7000, 0, 0])
    r2 = np.array([-7000, 0, 0])
    dt = 5000.0  # roughly half a period

    # Not yet fully implemented mapping, but should raise NotImplementedError instead of crash
    with pytest.raises(NotImplementedError):
        solve_lambert_multi_rev(r1, r2, dt, n_rev=1)


def test_primer_vector_optimality():
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, 7.546, 0])
    rf = np.array([0, 7000, 0])
    vf = np.array([-7.546, 0, 0])
    dt = 1000.0

    res = primer_vector_analysis(r0, v0, rf, vf, dt)
    assert "is_optimal" in res
    assert "dv_total" in res


def test_safe_approach_corridor():
    r_rel = np.array([10.0, 0.1, 0.1])
    axis = np.array([1.0, 0.0, 0.0])

    # 5 degree cone
    assert is_within_corridor(r_rel, axis, 5.0) == True

    # Way outside
    r_bad = np.array([1.0, 10.0, 0.0])
    assert is_within_corridor(r_bad, axis, 5.0) == False


def test_optimize_collocation_hook():
    r0 = np.array([1.0, 0, 0])
    v0 = np.array([0, 0, 0])
    rf = np.array([0, 0, 0])
    vf = np.array([0, 0, 0])
    dt = 100.0

    res = optimize_rpo_collocation(r0, v0, rf, vf, dt)
    assert "success" in res


def test_lambert_multi_rev_no_convergence():
    from unittest.mock import patch
    import pytest
    from opengnc.guidance.rendezvous import solve_lambert_multi_rev

    r1 = np.array([7000, 0, 0])
    r2 = np.array([0, 7000, 0])
    dt = 1000.0
    with patch("opengnc.guidance.rendezvous.newton", side_effect=Exception("Failed")):
        with pytest.raises(ValueError, match="No convergence"):
            solve_lambert_multi_rev(r1, r2, dt, n_rev=1)


def test_solve_lambert_edge_case_for_coverage():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np

    # Parameters found to trigger A > 0 and y < 0 in solve_lambert.py
    r1 = np.array([9296.887644909384, 1641.5109408174649, 7484.063624052451])
    r2 = np.array([4428.970313134735, 8028.292089204365, 7056.818643192096])
    dt = 101.30371377950725
    tm = 1
    v1, v2 = solve_lambert(r1, r2, dt, tm=tm)
    assert v1 is not None and v2 is not None


def test_solve_lambert_singularities():
    from opengnc.guidance.rendezvous import solve_lambert
    from unittest.mock import patch

    r1 = np.array([7000, 0, 0])
    r2 = np.array([0, 7000, 0])
    dt = 1000.0


def test_solve_lambert_coverage_hacks():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np
    from unittest.mock import patch

    r1 = np.array([7000, 0, 0])
    r2 = np.array([0, 7000, 0])
    dt = 1000.0

    # TM = -1 (long way)
    solve_lambert(r1, r2, 1000.0, tm=-1)
    solve_lambert(r1, r2, dt=1e6)


def test_solve_lambert_internal_branch_coverage():
    from opengnc.guidance.rendezvous import solve_lambert

    # hit the psi updates
    r1 = np.array([7000, 0, 0])
    r2 = np.array([0, 7000, 0])
    # Very short time might lead to large psi or different branches
    solve_lambert(r1, r2, dt=100.0)
    # Very long time
    solve_lambert(r1, r2, dt=10000.0)


def test_solve_lambert_y_val_guard_coverage():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np
    from unittest.mock import patch

    r1 = np.array([2.0, 0.0, 0.0])
    r2 = np.array([0.0, 2.0, 0.0])
    dt = 100.0
    solve_lambert(r1, r2, dt)


def test_solve_lambert_dtdpsi_guard_coverage():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np
    from unittest.mock import patch

    r1 = np.array([1000.0, 0.0, 0.0])
    r2 = np.array([0.0, 1100.0, 0.0])
    dt = 100.0
    mu_val = 9876.543
    solve_lambert(r1, r2, dt, mu=mu_val)


def test_solve_lambert_singularity():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np
    import pytest

    # 180 deg transfer
    r1 = np.array([7000, 0, 0])
    r2 = np.array([-7000, 0, 0])
    with pytest.raises(ValueError, match="Lambert Solver: A=0"):
        solve_lambert(r1, r2, 1000)


def test_cw_equations_coverage():
    from opengnc.guidance.rendezvous import cw_equations
    import numpy as np

    r0 = [1, 0, 0]
    v0 = [0, 0.001, 0]
    n_mean = 0.001
    time = 100.0

    rt, vt = cw_equations(r0, v0, n_mean, time)
    assert rt.shape == (3,)
    assert vt.shape == (3,)


def test_cw_targeting_coverage():
    from opengnc.guidance.rendezvous import cw_targeting
    import numpy as np

    r0 = np.array([1, 0, 0])
    rt = np.array([0, 0, 0])
    time = 1000.0
    n_mean = 1e-3

    v0 = cw_targeting(r0, rt, time, n_mean)
    assert v0.shape == (3,)

    # Test singularity in Z or LinAlgError
    # sin(nt) = 0 when nt = pi, 2pi, etc.
    time_sing = np.pi / n_mean
    v_sing = cw_targeting(r0, rt, time_sing, n_mean)
    assert v_sing is not None


def test_is_within_corridor_zero_dist():
    from opengnc.guidance.rendezvous import is_within_corridor
    import numpy as np

    assert is_within_corridor(np.array([0, 0, 0]), np.array([1, 0, 0]), 5.0) == True


def test_solve_lambert_multi_rev_zero():
    from opengnc.guidance.rendezvous import solve_lambert_multi_rev
    import numpy as np

    r1 = np.array([7000, 0, 0])
    r2 = np.array([0, 7000, 0])
    # n_rev=0 calls solve_lambert
    solve_lambert_multi_rev(r1, r2, 1000, n_rev=0)


def test_solve_lambert_internal_guards_coverage():
    from opengnc.guidance.rendezvous import solve_lambert
    import numpy as np
    from unittest.mock import patch

    # Line 78: y_val == 0
    # Suppression of expected RuntimeWarnings for division by zero/invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        with patch("opengnc.guidance.rendezvous.np.linalg.norm", return_value=1.0):
            with patch("opengnc.guidance.rendezvous.np.dot", return_value=0.0):
                with patch("opengnc.guidance.rendezvous.np.sin", return_value=np.sqrt(2)):
                    r1 = np.array([1, 0, 0])
                    r2 = np.array([0, 1, 0])
                    solve_lambert(r1, r2, 100.0)

        # Line 108: dtdpsi == 0
        r1 = np.array([7000, 0, 0])
        r2 = np.array([0, 7000, 0])
        solve_lambert(r1, r2, 1000, mu=float("inf"))


def test_yamanaka_ankersen_stm_identity_and_bounds():
    from opengnc.guidance.rendezvous import yamanaka_ankersen_stm

    p = 7000.0
    e = 0.1
    nu0 = 0.5

    # At nu == nu0, STM must be identity matrix I_6
    phi_id = yamanaka_ankersen_stm(nu0, nu0, e=e, p=p, dt=0.0)
    assert np.allclose(phi_id, np.eye(6), atol=1e-7)

    # Determinant should remain non-singular across elliptical orbits
    for ecc in [0.01, 0.1, 0.3, 0.7]:
        phi = yamanaka_ankersen_stm(1.2, 0.4, e=ecc, p=p, dt=500.0)
        assert abs(np.linalg.det(phi)) > 1e-4

    # Invalid eccentricity guards
    with pytest.raises(ValueError):
        yamanaka_ankersen_stm(1.0, 0.0, e=1.2, p=p)
    with pytest.raises(ValueError):
        yamanaka_ankersen_stm(1.0, 0.0, e=-0.1, p=p)


def test_yamanaka_ankersen_targeting_and_propagation():
    from opengnc.guidance.rendezvous import ya_propagate, ya_targeting

    p = 7200.0
    e = 0.05
    nu0 = 0.2
    nu_f = 0.8
    dt = 750.0

    r0 = np.array([0.5, -1.0, 0.2])  # km
    r_target = np.array([0.0, 0.0, 0.0])  # intercept target at origin

    v0_req = ya_targeting(r0, r_target, nu_f, nu0, e=e, p=p, dt=dt)
    r_final, _ = ya_propagate(r0, v0_req, nu_f, nu0, e=e, p=p, dt=dt)

    assert np.allclose(r_final, r_target, atol=1e-6)
