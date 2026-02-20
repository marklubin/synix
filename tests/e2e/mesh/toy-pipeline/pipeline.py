"""Toy pipeline for mesh E2E tests — Source -> Episodes -> Core + FlatFile."""

from synix import FlatFile, Pipeline, Source
from synix.transforms import CoreSynthesis, EpisodeSummary

source = Source("notes")
episodes = EpisodeSummary("episodes", depends_on=[source])
core = CoreSynthesis("core", depends_on=[episodes], context_budget=5000)

pipeline = Pipeline(
    "toy-mesh-test",
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
    FlatFile(
        "context",
        sources=[core],
        output_path="./build/context.md",
    ),
)
