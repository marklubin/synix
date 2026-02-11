# pipeline_monthly.py — Personal Memory Pipeline (Monthly Rollups)
#
# Usage:
#   cd examples/01-chatbot-export-synthesis
#   synix build pipeline_monthly.py
#
# Drop ChatGPT/Claude exports into ./exports/ before running.

from synix import Layer, Pipeline, Projection

pipeline = Pipeline("personal-memory")

pipeline.source_dir = "./exports"
pipeline.build_dir = "./build"

pipeline.llm_config = {
    "provider": "anthropic",
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

# Level 3: Core agent memory (uses Opus for highest-quality synthesis)
pipeline.add_layer(Layer(
    name="core",
    level=3,
    depends_on=["monthly"],
    transform="core_synthesis",
    grouping="single",
    context_budget=10000,
    config={
        "llm_config": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "max_tokens": 4096,
        },
    },
))

# --- Projections ---

# Search index — hierarchical search with provenance
pipeline.add_projection(Projection(
    name="memory-index",
    projection_type="search_index",
    sources=[
        {"layer": "episodes", "search": ["fulltext", "semantic"]},
        {"layer": "monthly", "search": ["fulltext", "semantic"]},
        {"layer": "core", "search": ["fulltext", "semantic"]},
    ],
    config={
        "embedding_config": {
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        },
    },
))

# Context document — ready-to-use agent system prompt
pipeline.add_projection(Projection(
    name="context-doc",
    projection_type="flat_file",
    sources=[{"layer": "core"}],
    config={"output_path": "./build/context.md"},
))
