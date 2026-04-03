"""Toy pipeline for knowledge server E2E tests.

Source → Episodes → Core + FlatFile (context-doc).
Minimal layers to exercise the full ingest → build → search → context flow.
"""

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.ext import CoreSynthesis, EpisodeSummary

source = Source("notes")
episodes = EpisodeSummary("episodes", depends_on=[source])
core = CoreSynthesis("core", depends_on=[episodes], context_budget=5000)

pipeline = Pipeline(
    "server-e2e-test",
    source_dir="./sources",
    build_dir="./build",
    llm_config={
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 512,
    },
)

pipeline.add(source, episodes, core)

pipeline.add(
    SearchIndex(
        "search",
        sources=[episodes, core],
        search=["fulltext"],
    )
)

pipeline.add(
    FlatFile(
        "context-doc",
        sources=[core],
        output_path="./build/context.md",
    ),
)
