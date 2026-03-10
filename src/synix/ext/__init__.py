"""Bundled opinionated transforms and migration compatibility exports.

Canonical ``synix.ext`` exports are the bundled memory-oriented transforms:

    from synix.ext import (
        CoreSynthesisTransform,
        EpisodeSummaryTransform,
        MonthlyRollupTransform,
        TopicalRollupTransform,
    )

The shorter aliases remain available for convenience:

    from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup, TopicalRollup

Generic platform transforms now live under ``synix.transforms``. They are
still re-exported here temporarily to preserve compatibility during the
migration.
"""

from synix.build.llm_transforms import (  # noqa: F401
    CoreSynthesis,
    CoreSynthesisTransform,
    EpisodeSummary,
    EpisodeSummaryTransform,
    MonthlyRollup,
    MonthlyRollupTransform,
    TopicalRollup,
    TopicalRollupTransform,
)
from synix.ext.chunk import Chunk  # noqa: F401
from synix.ext.fold_synthesis import FoldSynthesis  # noqa: F401
from synix.ext.group_synthesis import GroupSynthesis  # noqa: F401
from synix.ext.map_synthesis import MapSynthesis  # noqa: F401
from synix.ext.reduce_synthesis import ReduceSynthesis  # noqa: F401

__all__ = [
    "EpisodeSummary",
    "EpisodeSummaryTransform",
    "MonthlyRollup",
    "MonthlyRollupTransform",
    "TopicalRollup",
    "TopicalRollupTransform",
    "CoreSynthesis",
    "CoreSynthesisTransform",
    "MapSynthesis",
    "GroupSynthesis",
    "ReduceSynthesis",
    "FoldSynthesis",
    "Chunk",
]
