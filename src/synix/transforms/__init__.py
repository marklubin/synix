"""Public re-exports for generic platform transforms.

Canonical ``synix.transforms`` exports are the generic transform shapes:

    from synix.transforms import MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis, Merge

Bundled domain-specific memory transforms now live in ``synix.ext``. They are
still re-exported here temporarily to preserve compatibility during the
migration.
"""

from synix.build.llm_transforms import (  # noqa: F401
    CoreSynthesis,
    EpisodeSummary,
    MonthlyRollup,
    TopicalRollup,
)
from synix.build.merge_transform import Merge  # noqa: F401
from synix.ext.fold_synthesis import FoldSynthesis  # noqa: F401
from synix.ext.group_synthesis import GroupSynthesis  # noqa: F401
from synix.ext.map_synthesis import MapSynthesis  # noqa: F401
from synix.ext.reduce_synthesis import ReduceSynthesis  # noqa: F401

__all__ = [
    "MapSynthesis",
    "GroupSynthesis",
    "ReduceSynthesis",
    "FoldSynthesis",
    "Merge",
    "EpisodeSummary",
    "MonthlyRollup",
    "TopicalRollup",
    "CoreSynthesis",
]
