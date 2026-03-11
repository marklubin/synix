# pipeline.py — Agent Memory Pipeline
#
# Your agent accumulates knowledge across sessions. This pipeline turns
# session transcripts into searchable, structured memory with full provenance.
#
# Usage:
#   1. Save session transcripts as text/markdown files in ./sources/
#   2. uvx synix build
#   3. uvx synix release HEAD --to local
#   4. uvx synix search "your query" --release local
#
# From your agent at inference time:
#   import synix
#   project = synix.open_project(".")
#   mem = project.release("local")
#   results = mem.search("relevant context", limit=5)
#   context = mem.flat_file("context-doc")

from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

pipeline = Pipeline("agent-memory")

pipeline.source_dir = "./sources"

# --- LLM config ---
# Haiku is fast and cheap for episode summaries and rollups.
# For higher-quality core memory synthesis, override per-layer (see below).
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# --- Pipeline layers ---

# Level 0: Raw session transcripts
# Drop text/markdown files into ./sources/ — one file per session.
transcripts = Source("transcripts")

# Level 1: Episode summaries (one per session)
# Each session transcript → a structured episode summary.
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

# Level 2: Monthly rollups
# Episodes grouped by month → one summary per month.
# Change to TopicalRollup for theme-based clustering instead of calendar months:
#   from synix.ext import TopicalRollup
#   rollups = TopicalRollup("rollups", depends_on=[episodes],
#       uses=[memory_search], topics=["planning", "debugging", "decisions"])
monthly = MonthlyRollup("monthly", depends_on=[episodes])

# Level 3: Core memory
# All rollups → single evolving core memory document.
# This is what your agent loads as persistent context.
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

# --- Search ---

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext"],
    # Add semantic search (requires fastembed):
    #   modes=["fulltext", "semantic"],
    #   embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)

# --- Projections ---

# Search index — query from your agent via SDK, MCP, or CLI
pipeline.add(SynixSearch("search", surface=memory_search))

# Context document — load directly into your agent's system prompt
pipeline.add(
    FlatFile(
        "context-doc",
        sources=[core],
    )
)
