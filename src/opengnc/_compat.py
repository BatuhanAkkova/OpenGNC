"""
Compatibility helpers for transitional import paths.
"""

from __future__ import annotations

import warnings


def warn_legacy_import(old_path: str, new_path: str, remove_in: str = "2.0.0") -> None:
    """
    Emit a deprecation warning for a legacy import path.
    """
    warnings.warn(
        (
            f"`{old_path}` is deprecated and will be removed in OpenGNC {remove_in}. "
            f"Use `{new_path}` instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
