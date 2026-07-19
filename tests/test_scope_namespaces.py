import importlib
import warnings

from opengnc.experimental.edl import ballistic_entry_dynamics as experimental_ballistic_entry_dynamics
from opengnc.experimental.interfaces.gmat import GMATInterface as ExperimentalGMATInterface
from opengnc.ops.ground_segment.ccsds import SpacePacket as OpsSpacePacket


def test_legacy_ground_segment_import_warns_and_matches_new_class():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import opengnc.ground_segment as legacy_ground_segment

        importlib.reload(legacy_ground_segment)

    assert legacy_ground_segment.SpacePacket is OpsSpacePacket
    assert any("opengnc.ground_segment" in str(w.message) for w in caught)


def test_legacy_edl_import_warns_and_matches_new_function():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import opengnc.edl as legacy_edl

        importlib.reload(legacy_edl)

    assert legacy_edl.ballistic_entry_dynamics is experimental_ballistic_entry_dynamics
    assert any("opengnc.edl" in str(w.message) for w in caught)


def test_legacy_gmat_import_warns_and_matches_new_class():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import opengnc.interfaces.gmat.gmat_interface as legacy_gmat

        importlib.reload(legacy_gmat)

    assert legacy_gmat.GMATInterface is ExperimentalGMATInterface
    assert any("opengnc.interfaces.gmat.gmat_interface" in str(w.message) for w in caught)


def test_legacy_submodule_class_identity_is_preserved():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from opengnc.ground_segment.ccsds import SpacePacket

    assert SpacePacket is OpsSpacePacket


def test_legacy_dashboard_server_import_warns_and_matches_new_symbol():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import opengnc.dashboard.server as legacy_dashboard

        importlib.reload(legacy_dashboard)

    from opengnc.ops.dashboard.server import run_server as ops_run_server

    assert legacy_dashboard.run_server is ops_run_server
    assert any("opengnc.dashboard.server" in str(w.message) for w in caught)
