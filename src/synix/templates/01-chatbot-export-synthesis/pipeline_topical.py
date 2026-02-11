# pipeline_topical.py — Personal Memory Pipeline (Topic-Based Rollups)
#
# Alternate pipeline — episodes grouped by topic instead of month.
# Run after pipeline_monthly.py to demo incremental rebuild:
# transcripts + episodes stay cached, only rollups + core rebuild.
#
# Usage:
#   cd examples/01-chatbot-export-synthesis
#   synix build pipeline_topical.py

from synix import Layer, Pipeline, Projection

pipeline = Pipeline("personal-memory-topical")

pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"

pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# --- Layers ---

# Level 0: Same raw transcripts (CACHED from first run)
pipeline.add_layer(Layer(
    name="transcripts",
    level=0,
    transform="parse",
))

# Level 1: Same episode summaries (CACHED from first run)
pipeline.add_layer(Layer(
    name="episodes",
    level=1,
    depends_on=["transcripts"],
    transform="episode_summary",
    grouping="by_conversation",
))

# Level 2: CHANGED — Topic-based rollups instead of monthly
# LLM clusters episodes by topic, then synthesizes each cluster
pipeline.add_layer(Layer(
    name="topics",
    level=2,
    depends_on=["episodes"],
    transform="topical_rollup",
    grouping="by_topic",
    config={
        "topics": [
            "programming-and-tools",
            "ai-and-llms",
            "systems-and-infrastructure",
            "debugging-and-testing",
            "general",
        ],
    },
))

# Level 3: Core memory — same transform, but new inputs (topics instead of monthly)
# Will REBUILD because its dependency changed
# For production use, upgrade to Opus for highest-quality synthesis:
#   config={"llm_config": {"model": "claude-opus-4-6", "max_tokens": 4096}}
pipeline.add_layer(Layer(
    name="core",
    level=3,
    depends_on=["topics"],
    transform="core_synthesis",
    grouping="single",
    context_budget=10000,
))

# --- Projections ---

pipeline.add_projection(Projection(
    name="memory-index",
    projection_type="search_index",
    sources=[
        {"layer": "episodes", "search": ["fulltext", "semantic"]},
        {"layer": "topics", "search": ["fulltext", "semantic"]},
        {"layer": "core", "search": ["fulltext", "semantic"]},
    ],
    config={
        "embedding_config": {
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        },
    },
))

pipeline.add_projection(Projection(
    name="context-doc",
    projection_type="flat_file",
    sources=[{"layer": "core"}],
    config={"output_path": "./build/context.md"},
))
