"""
Deprecated compatibility wrapper for the experimental GMAT namespace.
"""

from opengnc._compat import warn_legacy_import
from opengnc.interfaces.gmat.gmat_interface import GMATInterface

warn_legacy_import("opengnc.interfaces.gmat", "opengnc.experimental.interfaces.gmat")

__all__ = ["GMATInterface"]
