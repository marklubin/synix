"""Claude Code session pipeline — 4-layer DAG for mesh distributed builds.

Source -> EpisodeSummary -> MonthlyRollup -> CoreSynthesis + SearchSurface + SynixSearch + FlatFile

This pipeline is designed for use with `synix mesh`. Claude Code sessions
(JSONL files from ~/.claude/projects) are submitted via the mesh client,
built on the server, and deployed back to clients as context documents.
"""

from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

# Define layers
source = Source("sessions")
summaries = EpisodeSummary("summaries", depends_on=[source])
rollups = MonthlyRollup("rollups", depends_on=[summaries])
core = CoreSynthesis("core", depends_on=[rollups], context_budget=15000)
session_search = SearchSurface("session-search", sources=[summaries, rollups, core], modes=["fulltext"])

# Build pipeline
pipeline = Pipeline(
    "claude-sessions",
    source_dir="./sources",
    build_dir="./build",
    llm_config={
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 2048,
    },
)

pipeline.add(source, summaries, rollups, core, session_search)

pipeline.add(
    SynixSearch("search", surface=session_search),
    FlatFile(
        "context",
        sources=[core],
        output_path="./build/context.md",
    ),
)
