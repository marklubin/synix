# pipeline_topical.py — Personal Memory Pipeline (Topic-Based Rollups)
#
# Alternate pipeline — episodes grouped by topic instead of month.
# Run after pipeline_monthly.py to demo incremental rebuild:
# transcripts + episodes stay cached, only rollups + core rebuild.
#
# Usage:
#   cd templates/01-chatbot-export-synthesis
#   uvx synix build pipeline_topical.py

from synix import FlatFile, Pipeline, SearchIndex, SearchSurface, Source
from synix.ext import CoreSynthesis, EpisodeSummary, TopicalRollup

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
transcripts = Source("transcripts")

# Level 1: Same episode summaries (CACHED from first run)
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

# Build-time search surface used by the topical rollup
episode_search = SearchSurface(
    "episode-search",
    sources=[episodes],
    modes=["fulltext"],
)

# Level 2: CHANGED — Topic-based rollups instead of monthly
# LLM clusters episodes by topic, then synthesizes each cluster using
# the declared episode search surface instead of the global search.db path.
topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    uses=[episode_search],
    config={
        "topics": [
            "programming-and-tools",
            "ai-and-llms",
            "systems-and-infrastructure",
            "debugging-and-testing",
            "general",
        ],
    },
)

# Level 3: Core memory — same transform, but new inputs (topics instead of monthly)
# Will REBUILD because its dependency changed
# For production use, upgrade to Opus for highest-quality synthesis:
#   config={"llm_config": {"model": "claude-opus-4-6", "max_tokens": 4096}}
core = CoreSynthesis("core", depends_on=[topics], context_budget=10000)

pipeline.add(transcripts, episodes, episode_search, topics, core)

# --- Projections ---

pipeline.add(
    SearchIndex(
        "memory-index",
        sources=[episodes, topics, core],
        search=["fulltext", "semantic"],
        embedding_config={
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        },
    )
)

pipeline.add(
    FlatFile(
        "context-doc",
        sources=[core],
        output_path="./build/context.md",
    )
)
