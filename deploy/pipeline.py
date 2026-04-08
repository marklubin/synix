"""Knowledge server pipeline — agent-driven memory architecture.

Memory types:
  Experiential (live): sessions + exports → episodes → fold through 4 lenses
  Experiential (backfill): old exports → episodes → search only (no fold)
  Semantic: documents + reports → pass through → fold through 4 lenses
  Reference: background material → chunk → search only

Analysis lenses (L2 fold):
  builder    — what's being built, shipped, decided technically
  researcher — what's being investigated, learned, compared
  operator   — what needs attention, what's blocked, what's due
  ideator    — ideas, what-ifs, speculative directions, connections

Synthesis (L3 reduce):
  working_memory — builder + operator → what's active now
  idea_garden    — ideator → seeds, connections, possibilities
  self_model     — all 4 lenses → identity, preferences, patterns
  core_memory    — everything → stable context for agent injection

Context injection = core_memory + self_model + working_memory + idea_garden
                  + search tools for episodes + reference
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
builder = _agents["builder"]
researcher = _agents["researcher"]
operator = _agents["operator"]
ideator = _agents["ideator"]
synthesizer = _agents["synthesizer"]

_t = lambda key, fallback="": _store.get(key) or fallback  # noqa: E731

# ---------------------------------------------------------------------------
# Sources (6 buckets)
# ---------------------------------------------------------------------------

# Experiential — live (feeds folds)
sessions = Source("sessions", dir="./sources/sessions")
exports = Source("exports", dir="./sources/exports")

# Experiential — backfill (search only, no fold)
backfill = Source("backfill", dir="./sources/backfill")

# Semantic (feeds folds)
documents = Source("documents", dir="./sources/documents")
reports = Source("reports", dir="./sources/reports")

# Reference (search only)
reference = Source("reference", dir="./sources/reference")

# ---------------------------------------------------------------------------
# Level 1 — Compression
# ---------------------------------------------------------------------------

# Live episodes (sessions + exports → summarizer)
live_episodes = MapSynthesis(
    "live-episodes",
    depends_on=[sessions, exports],
    prompt=_t("task-episode", "Summarize this conversation:\n\n{artifact}"),
    agent=summarizer,
    artifact_type="episode",
)

# Backfill episodes (old exports → same summarizer, but not folded)
backfill_episodes = MapSynthesis(
    "backfill-episodes",
    depends_on=[backfill],
    prompt=_t("task-episode", "Summarize this conversation:\n\n{artifact}"),
    agent=summarizer,
    artifact_type="episode",
)

# Reference chunks (no LLM)
reference_chunks = Chunk(
    "reference-chunks",
    depends_on=[reference],
    separator="\n\n",
    chunk_size=512,
    chunk_overlap=50,
)

# ---------------------------------------------------------------------------
# Level 2 — Analysis lenses (fold — live content only)
# ---------------------------------------------------------------------------

# Content pool: live episodes + semantic docs (NOT backfill)
_content_pool = [live_episodes, documents, reports]

builder_lens = FoldSynthesis(
    "builder",
    depends_on=_content_pool,
    sort_by="date",
    label="builder",
    artifact_type="lens",
    prompt=_t("task-builder"),
    agent=builder,
    initial="# Builder\n\nNo activity yet.",
)

researcher_lens = FoldSynthesis(
    "researcher",
    depends_on=_content_pool,
    sort_by="date",
    label="researcher",
    artifact_type="lens",
    prompt=_t("task-researcher"),
    agent=researcher,
    initial="# Researcher\n\nNo activity yet.",
)

operator_lens = FoldSynthesis(
    "operator",
    depends_on=_content_pool,
    sort_by="date",
    label="operator",
    artifact_type="lens",
    prompt=_t("task-operator"),
    agent=operator,
    initial="# Operator\n\nNo activity yet.",
)

ideator_lens = FoldSynthesis(
    "ideator",
    depends_on=_content_pool,
    sort_by="date",
    label="ideator",
    artifact_type="lens",
    prompt=_t("task-ideator"),
    agent=ideator,
    initial="# Ideator\n\nNo activity yet.",
)

# ---------------------------------------------------------------------------
# Level 3 — Synthesis
# ---------------------------------------------------------------------------

working_memory = ReduceSynthesis(
    "working-memory",
    depends_on=[builder_lens, operator_lens],
    prompt=_t("task-working-memory"),
    agent=synthesizer,
    label="working-memory",
    artifact_type="synthesis",
)

idea_garden = ReduceSynthesis(
    "idea-garden",
    depends_on=[ideator_lens],
    prompt=_t("task-idea-garden"),
    agent=synthesizer,
    label="idea-garden",
    artifact_type="synthesis",
)

self_model = ReduceSynthesis(
    "self-model",
    depends_on=[builder_lens, researcher_lens, operator_lens, ideator_lens],
    prompt=_t("task-self-model"),
    agent=synthesizer,
    label="self-model",
    artifact_type="synthesis",
)

core_memory = ReduceSynthesis(
    "core-memory",
    depends_on=[working_memory, idea_garden, self_model],
    prompt=_t("task-core-memory"),
    agent=synthesizer,
    label="core-memory",
    artifact_type="core_memory",
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
pipeline.add(sessions, exports, backfill, documents, reports, reference)

# L1 — compression
pipeline.add(live_episodes, backfill_episodes, reference_chunks)

# L2 — analysis lenses
pipeline.add(builder_lens, researcher_lens, operator_lens, ideator_lens)

# L3 — synthesis
pipeline.add(working_memory, idea_garden, self_model, core_memory)

# Search surfaces
main_surface = SearchSurface(
    "main",
    sources=[
        core_memory, working_memory, idea_garden, self_model,
        builder_lens, researcher_lens, operator_lens, ideator_lens,
    ],
    modes=["fulltext", "semantic"],
    embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)

episodes_surface = SearchSurface(
    "episodes",
    sources=[live_episodes, backfill_episodes],
    modes=["fulltext"],
)

reference_surface = SearchSurface(
    "reference",
    sources=[reference_chunks],
    modes=["fulltext"],
)

pipeline.add(main_surface, episodes_surface, reference_surface)
pipeline.add(SynixSearch("search", surface=main_surface))
pipeline.add(SynixSearch("episode-search", surface=episodes_surface))
pipeline.add(SynixSearch("reference-search", surface=reference_surface))

# Context injection — what the agent gets as system context
pipeline.add(FlatFile("context-doc", sources=[core_memory, self_model, working_memory, idea_garden]))
