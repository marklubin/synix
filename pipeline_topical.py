# pipeline_topical.py — Personal Memory Pipeline (Topic-Based Rollups)
# Alternate pipeline for the demo "config change" moment
# Same data, different architecture — episodes grouped by topic instead of month

from synix import Pipeline, Layer, Projection

pipeline = Pipeline("personal-memory-topical")

pipeline.source_dir = "./exports"
pipeline.build_dir = "./build"

pipeline.llm_config = {
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
            "career-and-job-search",
            "technical-projects",
            "san-francisco",
            "personal-growth",
            "ai-and-agents",
        ],
        # At runtime, the topical rollup transform queries the episode
        # search index (already built) to find relevant episodes per topic.
        # This scales to 1000+ conversations without blowing up context.
    },
))

# Level 3: Core memory — same transform, but new inputs (topics instead of monthly)
# Will REBUILD because its dependency changed
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
        {"layer": "episodes", "search": ["fulltext"]},
        {"layer": "topics", "search": ["fulltext"]},
        {"layer": "core", "search": ["fulltext"]},
    ],
))

pipeline.add_projection(Projection(
    name="context-doc",
    projection_type="flat_file",
    sources=[{"layer": "core"}],
    config={"output_path": "./build/context.md"},
))
