"""
Deprecated compatibility exports for operational dashboard modules.
"""

from opengnc._compat import warn_legacy_import
from opengnc.ops.dashboard import run_server

warn_legacy_import("opengnc.dashboard", "opengnc.ops.dashboard")

__all__ = ["run_server"]
