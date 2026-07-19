"""
Deprecated compatibility wrapper for the experimental GMAT interface.
"""

from opengnc._compat import warn_legacy_import
from opengnc.experimental.interfaces.gmat import GMATInterface

warn_legacy_import(
    "opengnc.interfaces.gmat.gmat_interface",
    "opengnc.experimental.interfaces.gmat",
)

__all__ = ["GMATInterface"]
