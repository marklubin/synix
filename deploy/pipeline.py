"""Knowledge server pipeline — agent-driven.

5 buckets → content pool → 5 fold layers → core + work-status → search + projections

All transforms use named agents. Agent instructions (persona/semantics) and task
prompts (structure/placeholders) are both managed in the PromptStore, editable
via the viewer's Prompts tab.

Agents:
  summarizer  — episode summaries from conversations (map)
  writer      — short-window, long-window, work-status (fold)
  tracker     — research threads, open threads, user model (fold)
  synthesizer — core memory from rollups (reduce)
"""

from __future__ import annotations

from pathlib import Path

from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.server.prompt_store import PromptStore
from synix.transforms import Chunk, FoldSynthesis, MapSynthesis, ReduceSynthesis
from synix.workspace import load_agents as _load_agents

# ---------------------------------------------------------------------------
# Prompt + Agent setup
# ---------------------------------------------------------------------------

_here = Path(__file__).parent
_store = PromptStore(_here / ".synix" / "prompts.db")
_store.seed_from_files(_here / "prompts")

_agents = _load_agents(str(_here / "synix.toml"))

summarizer = _agents["summarizer"]
writer = _agents["writer"]
tracker = _agents["tracker"]
synthesizer = _agents["synthesizer"]

# Task prompts from PromptStore (editable in viewer)
_t = lambda key, fallback="": _store.get(key) or fallback  # noqa: E731

# ---------------------------------------------------------------------------
# Sources (5 buckets)
# ---------------------------------------------------------------------------

sessions = Source("sessions", dir="./sources/sessions")
exports = Source("exports", dir="./sources/exports")
documents = Source("documents", dir="./sources/documents")
reports = Source("reports", dir="./sources/reports")
reference = Source("reference", dir="./sources/reference")

# ---------------------------------------------------------------------------
# Level 1 — Episode summaries (sessions + exports → summarizer agent)
# ---------------------------------------------------------------------------

episodes = MapSynthesis(
    "episodes",
    depends_on=[sessions, exports],
    prompt=_t("task-episode", "Summarize this conversation:\n\n{artifact}"),
    agent=summarizer,
    artifact_type="episode",
)

# ---------------------------------------------------------------------------
# Level 2 — Rollups (content pool → fold agents)
# ---------------------------------------------------------------------------

_content_pool = [episodes, documents, reports]

short_window = FoldSynthesis(
    "short-window",
    depends_on=_content_pool,
    sort_by="date",
    label="short-window",
    artifact_type="rollup",
    prompt=_t("task-short-window"),
    agent=writer,
    initial="# Recent Activity\n\nNo recent activity recorded.",
)

long_window = FoldSynthesis(
    "long-window",
    depends_on=_content_pool,
    sort_by="date",
    label="long-window",
    artifact_type="rollup",
    prompt=_t("task-long-window"),
    agent=writer,
    initial="# Long-Window Summary\n\nNo activity recorded yet.",
)

research_threads = FoldSynthesis(
    "research",
    depends_on=_content_pool,
    sort_by="date",
    label="research-threads",
    artifact_type="rollup",
    prompt=_t("task-research"),
    agent=tracker,
    initial="# Research Threads\n\nNo active research threads.",
)

open_threads = FoldSynthesis(
    "open-threads",
    depends_on=_content_pool,
    sort_by="date",
    label="open-threads",
    artifact_type="rollup",
    prompt=_t("task-open-threads"),
    agent=tracker,
    initial="# Open Threads\n\nNo open items.",
)

user_model = FoldSynthesis(
    "user-model",
    depends_on=_content_pool,
    sort_by="date",
    label="user-model",
    artifact_type="rollup",
    prompt=_t("task-user-model"),
    agent=tracker,
    initial="# User Model\n\nNo observations yet.",
)

# ---------------------------------------------------------------------------
# Level 3 — Synthesis
# ---------------------------------------------------------------------------

core = ReduceSynthesis(
    "core",
    depends_on=[long_window, research_threads, open_threads, user_model],
    prompt=_t("task-core", "Synthesize these documents into a single core memory document:\n\n{artifacts}"),
    agent=synthesizer,
    label="core-memory",
    artifact_type="core_memory",
)

work_status = FoldSynthesis(
    "work-status",
    depends_on=[short_window, long_window, open_threads],
    sort_by="date",
    label="work-status",
    artifact_type="report",
    prompt=_t("task-work-status"),
    agent=writer,
    initial="# Work Status\n\nNo status information yet.",
)

# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

pipeline = Pipeline(
    "knowledge-server",
    source_dir="./sources",
    build_dir="./build",
    llm_config={
        "provider": "openai-compatible",
        "model": "Qwen/Qwen3.5-2B",
        "base_url": "http://localhost:8100/v1",
        "api_key": "not-needed",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
)

# Sources
pipeline.add(sessions, exports, documents, reports, reference)

# Level 1
pipeline.add(episodes)

# Level 2 — rollups
pipeline.add(short_window, long_window, research_threads, open_threads, user_model)

# Level 3 — synthesis
pipeline.add(core, work_status)

# Chunk reference docs for search
reference_chunks = Chunk(
    "reference-chunks",
    depends_on=[reference],
    separator="\n\n",
    chunk_size=512,
    chunk_overlap=50,
)
pipeline.add(reference_chunks)

# Search surfaces
main_surface = SearchSurface(
    "main",
    sources=[core, work_status, user_model, short_window, long_window, research_threads, open_threads],
    modes=["fulltext", "semantic"],
    embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)

reference_surface = SearchSurface(
    "reference",
    sources=[reference_chunks],
    modes=["fulltext"],
)

pipeline.add(main_surface, reference_surface)
pipeline.add(SynixSearch("search", surface=main_surface))
pipeline.add(SynixSearch("reference-search", surface=reference_surface))

# Flat file projections
pipeline.add(FlatFile("context-doc", sources=[core, work_status, user_model]))
pipeline.add(FlatFile("weekly-brief", sources=[short_window, open_threads]))
