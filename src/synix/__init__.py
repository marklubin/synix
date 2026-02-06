"""Synix - A declarative pipeline system for processing AI conversation exports.

Usage:
    from synix import Pipeline

    pipeline = Pipeline("personal-memory", agent="mark")
    pipeline.source("claude", file="~/exports/claude.json", format="claude-export")
    pipeline.transform("summaries", from_="claude", prompt=summarize)
    pipeline.aggregate("monthly", from_="summaries", period="month", prompt=reflect)
    pipeline.fold("narrative", from_="monthly", prompt=evolve)
    pipeline.merge("combined", sources=["a", "b"], prompt=combine)
    pipeline.artifact("report", from_="narrative", surface="file://output/report.md")
    pipeline.run()
"""

from synix.db.artifacts import Record, RecordSource
from synix.pipeline import Pipeline, PlanResult, RunResult
from synix.services.search import SearchHit
from synix.steps.aggregate import AggregateStep
from synix.steps.fold import FoldStep
from synix.steps.merge import MergeStep
from synix.steps.transform import TransformStep
from synix.surfaces.base import PublishResult, Surface
from synix.surfaces.file import FileSurface

__all__ = [
    "AggregateStep",
    "FileSurface",
    "FoldStep",
    "MergeStep",
    "Pipeline",
    "PlanResult",
    "PublishResult",
    "Record",
    "RecordSource",
    "RunResult",
    "SearchHit",
    "Surface",
    "TransformStep",
]

__version__ = "0.1.0"
