# pipeline.py — Personal Memory Pipeline (Monthly Rollups)
# This is the default pipeline for the demo

from synix import Pipeline, Layer, Projection

pipeline = Pipeline("personal-memory")

pipeline.source_dir = "./exports"
pipeline.build_dir = "./build"

pipeline.llm_config = {
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# --- Layers ---

# Level 0: Raw conversation transcripts
pipeline.add_layer(Layer(
    name="transcripts",
    level=0,
    transform="parse",
))

# Level 1: Episode summaries (one per conversation)
pipeline.add_layer(Layer(
    name="episodes",
    level=1,
    depends_on=["transcripts"],
    transform="episode_summary",
    grouping="by_conversation",
))

# Level 2: Monthly rollups
pipeline.add_layer(Layer(
    name="monthly",
    level=2,
    depends_on=["episodes"],
    transform="monthly_rollup",
    grouping="by_month",
))

# Level 3: Core agent memory
pipeline.add_layer(Layer(
    name="core",
    level=3,
    depends_on=["monthly"],
    transform="core_synthesis",
    grouping="single",
    context_budget=10000,
))

# --- Projections ---

# Search index — hierarchical search with provenance
pipeline.add_projection(Projection(
    name="memory-index",
    projection_type="search_index",
    sources=[
        {"layer": "episodes", "search": ["fulltext"]},
        {"layer": "monthly", "search": ["fulltext"]},
        {"layer": "core", "search": ["fulltext"]},
    ],
))

# Context document — ready-to-use agent system prompt
pipeline.add_projection(Projection(
    name="context-doc",
    projection_type="flat_file",
    sources=[{"layer": "core"}],
    config={"output_path": "./build/context.md"},
))
