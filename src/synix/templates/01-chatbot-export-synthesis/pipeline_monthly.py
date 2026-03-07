# pipeline_monthly.py — Personal Memory Pipeline (Monthly Rollups)
#
# Usage:
#   cd templates/01-chatbot-export-synthesis
#   uvx synix build pipeline_monthly.py
#
# Drop ChatGPT/Claude exports into ./sources/ before running.

from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

pipeline = Pipeline("personal-memory")

pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"

pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# --- Layers ---

# Level 0: Raw conversation transcripts
transcripts = Source("transcripts")

# Level 1: Episode summaries (one per conversation)
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

# Level 2: Monthly rollups
monthly = MonthlyRollup("monthly", depends_on=[episodes])

# Level 3: Core agent memory
# For production use, upgrade to Opus for highest-quality synthesis:
#   config={"llm_config": {"model": "claude-opus-4-6", "max_tokens": 4096}}
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext", "semantic"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)

# --- Projections ---

# SynixSearch — default local search output over the declared memory surface
pipeline.add(SynixSearch("search", surface=memory_search))

# Context document — ready-to-use agent system prompt
pipeline.add(
    FlatFile(
        "context-doc",
        sources=[core],
        output_path="./build/context.md",
    )
)
