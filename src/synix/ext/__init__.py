"""Configurable transform library for common pipeline patterns.

Usage::

    from synix.ext import MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis
"""

from synix.ext.fold_synthesis import FoldSynthesis  # noqa: F401
from synix.ext.group_synthesis import GroupSynthesis  # noqa: F401
from synix.ext.map_synthesis import MapSynthesis  # noqa: F401
from synix.ext.reduce_synthesis import ReduceSynthesis  # noqa: F401
