"""
Deprecated compatibility wrapper for the operational dashboard server.
"""

from opengnc._compat import warn_legacy_import
from opengnc.ops.dashboard.server import app, run_server

warn_legacy_import("opengnc.dashboard.server", "opengnc.ops.dashboard.server")

__all__ = ["app", "run_server"]
