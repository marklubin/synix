"""Shared utilities for ext transforms."""

from __future__ import annotations

import inspect
from collections.abc import Callable


def stable_callable_repr(fn: Callable) -> str:
    """Return a stable string representation of a callable for cache keys.

    Uses inspect.getsource (deterministic across process restarts) with
    repr() fallback for builtins/C extensions where source is unavailable.
    """
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
