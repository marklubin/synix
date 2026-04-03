"""Knowledge server pipeline.

5 buckets → content pool → 5 fold layers → core + work-status → search + projections

Sessions and exports go through EpisodeSummary.
Documents and reports pass through as-is.
Reference bypasses everything — search only.

All rollup layers are FoldSynthesis — same mechanism, differentiated by prompt.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.adapters.registry import parse_claude_code, register_adapter
from synix.core.models import Artifact
from synix.transforms import Chunk, CoreSynthesis, EpisodeSummary, FoldSynthesis

# ---------------------------------------------------------------------------
# Adapters — session/export format handling
# ---------------------------------------------------------------------------

_INVALID_JSON_CHARS = re.compile(r"[\ud800-\udfff\x00-\x08\x0b\x0c\x0e-\x1f]")
_SUBSESSION_RE = re.compile(r"^(.+)_sub\d{4}\.jsonl\.gz$")


def _sanitize_for_json(text: str) -> str:
    return _INVALID_JSON_CHARS.sub("", text)


def _sanitize_artifacts(artifacts: list[Artifact]) -> list[Artifact]:
    for a in artifacts:
        sanitized = _sanitize_for_json(a.content)
        if sanitized != a.content:
            a.content = sanitized
            a.artifact_id = f"sha256:{hashlib.sha256(sanitized.encode()).hexdigest()}"
    return artifacts


@register_adapter([".jsonl.gz"])
def parse_claude_code_gz(filepath: str | Path) -> list[Artifact]:
    """Decompress, merge subsessions, sanitize, parse."""
    filepath = Path(filepath)
    if _SUBSESSION_RE.match(filepath.name):
        return []

    base_stem = filepath.name.removesuffix(".jsonl.gz")
    parent = filepath.parent
    files = []
    base = parent / f"{base_stem}.jsonl.gz"
    if base.exists():
        files.append(base)
    files.extend(sorted(parent.glob(f"{base_stem}_sub*.jsonl.gz")))

    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / f"{base_stem}.jsonl"
    try:
        with open(tmp_path, "wb") as out:
            for gz_file in files:
                with gzip.open(gz_file, "rb") as gz:
                    shutil.copyfileobj(gz, out)
        return _sanitize_artifacts(parse_claude_code(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
        Path(tmp_dir).rmdir()


@register_adapter([".jsonl"])
def parse_claude_code_sanitized(filepath: str | Path) -> list[Artifact]:
    return _sanitize_artifacts(parse_claude_code(filepath))


# ---------------------------------------------------------------------------
# Sources (5 buckets)
# ---------------------------------------------------------------------------

sessions = Source("sessions", dir="./sources/sessions")
exports = Source("exports", dir="./sources/exports")
documents = Source("documents", dir="./sources/documents")
reports = Source("reports", dir="./sources/reports")
reference = Source("reference", dir="./sources/reference")

# ---------------------------------------------------------------------------
# Level 1 — Episode Summary (sessions + exports only)
# ---------------------------------------------------------------------------

episodes = EpisodeSummary(
    "episodes",
    depends_on=[sessions, exports],
)

# ---------------------------------------------------------------------------
# Level 2 — Rollups (all FoldSynthesis, same mechanism, different prompts)
#
# Each reads from the content pool: episodes + documents + reports
# Reference is NOT in the content pool — it bypasses to search only.
# ---------------------------------------------------------------------------

# Content pool sources for rollups
_content_pool = [episodes, documents, reports]

short_window = FoldSynthesis(
    "short-window",
    depends_on=_content_pool,
    sort_by="date",
    label="short-window",
    artifact_type="rollup",
    prompt="""\
You are maintaining a short-window status document — what's been happening in the last few days.

Current status:
{accumulated}

New content (label: {label}):
{artifact}

Update the status to incorporate this. Focus on:
- What was worked on in the last few days
- Decisions made, outcomes reached
- Conversations had, people talked to
- Current momentum and direction

Drop anything older than ~a week. Keep it 200-400 words, factual, status-report style. \
Use specific project names, people, and outcomes.""",
    initial="# Recent Activity\n\nNo recent activity recorded.",
)

long_window = FoldSynthesis(
    "long-window",
    depends_on=_content_pool,
    sort_by="date",
    label="long-window",
    artifact_type="rollup",
    prompt="""\
You are maintaining a long-window summary — trends and trajectory over the last couple months.

Current summary:
{accumulated}

New content (label: {label}):
{artifact}

Update the summary. Focus on:
- Recurring themes and patterns across weeks
- How projects and priorities have evolved
- Strategic direction and trajectory shifts
- Relationships and collaborations developing

Drop granular daily details — keep the arc, not the events. Compress older material \
into higher-level observations. 300-500 words.""",
    initial="# Long-Window Summary\n\nNo activity recorded yet.",
)

research_threads = FoldSynthesis(
    "research",
    depends_on=_content_pool,
    sort_by="date",
    label="research-threads",
    artifact_type="rollup",
    prompt="""\
You are maintaining a research thread tracker.

Current research state:
{accumulated}

New content (label: {label}):
{artifact}

If this contains research activity — hypotheses, experiments, literature review, \
technical investigation, architectural exploration, analysis — update the relevant \
thread or create a new one.

If it's not research-related, return the current state unchanged.

For each thread, maintain:
- **Question/Hypothesis**: what's being investigated
- **Status**: ACTIVE / VALIDATED / REFUTED / DORMANT
- **Evidence**: key findings, data points, results
- **Open questions**: what's still unknown

Keep it structured. One thread per section.""",
    initial="# Research Threads\n\nNo active research threads.",
)

open_threads = FoldSynthesis(
    "open-threads",
    depends_on=_content_pool,
    sort_by="date",
    label="open-threads",
    artifact_type="rollup",
    prompt="""\
You are maintaining a list of open threads — unresolved items, blockers, \
pending actions, and follow-ups.

Current open threads:
{accumulated}

New content (label: {label}):
{artifact}

Update:
- Add new unresolved items, blockers, or pending actions
- Remove or mark items that this content resolves
- Keep each item as one line with what it's waiting on

Format as a clean bullet list. Be specific about what's pending and who/what \
it's blocked on. Drop items that seem stale (no mention in weeks).""",
    initial="# Open Threads\n\nNo open items.",
)

user_model = FoldSynthesis(
    "user-model",
    depends_on=_content_pool,
    sort_by="date",
    label="user-model",
    artifact_type="rollup",
    prompt="""\
You are building and maintaining a model of the user — hypotheses about what \
works for them, what doesn't, their preferences, working style, and patterns.

Current user model:
{accumulated}

New content (label: {label}):
{artifact}

Look for signals about:
- **Working style**: how they prefer to collaborate, communicate, make decisions
- **What works**: approaches, tools, framings that they respond well to
- **What doesn't work**: things that frustrate them, waste time, or miss the mark
- **Preferences**: technical opinions, aesthetic choices, recurring standards
- **Patterns**: habits, rhythms, tendencies that repeat across sessions

Update the model. Add new observations. Refine existing hypotheses with new evidence. \
Mark things you're less sure about. If nothing in this content is relevant to the \
user model, return the current state unchanged.

Write as structured observations, not a narrative.""",
    initial="# User Model\n\nNo observations yet.",
)

# ---------------------------------------------------------------------------
# Level 3 — Synthesis
# ---------------------------------------------------------------------------

core = CoreSynthesis(
    "core",
    depends_on=[long_window, research_threads, open_threads, user_model],
    context_budget=10000,
)

work_status = FoldSynthesis(
    "work-status",
    depends_on=[short_window, long_window, open_threads],
    sort_by="date",
    label="work-status",
    artifact_type="report",
    prompt="""\
You are generating a structured work status report.

Current report:
{accumulated}

New input:
{artifact}

Maintain these sections:

## Active Projects
For each: status (active/stalled/wrapping up), recent work, next steps.

## Recently Completed
Projects or tasks finished recently.

## Blockers & Open Questions
Unresolved issues, decisions needed, external dependencies.

Be specific about project names, tools, outcomes. Concise and actionable.""",
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
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
)

# Sources
pipeline.add(sessions, exports, documents, reports, reference)

# Level 1
pipeline.add(episodes)

# Level 2 — rollups (read from content pool)
pipeline.add(short_window, long_window, research_threads, open_threads, user_model)

# Level 3 — synthesis
pipeline.add(core, work_status)

# Chunk reference docs for useful semantic search
reference_chunks = Chunk(
    "reference-chunks",
    depends_on=[reference],
    strategy="separator",
    separator="\n\n",
    max_tokens=512,
    overlap=50,
)
pipeline.add(reference_chunks)

# Search — rollups + synthesis (default), reference chunks (opt-in)
# Raw episodes and documents are NOT in search by default.
main_surface = SearchSurface(
    "main",
    sources=[
        core, work_status, user_model,
        short_window, long_window, research_threads, open_threads,
    ],
    modes=["fulltext", "semantic"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
)

reference_surface = SearchSurface(
    "reference",
    sources=[reference_chunks],
    modes=["fulltext", "semantic"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
)

pipeline.add(main_surface, reference_surface)
pipeline.add(SynixSearch("search", surface=main_surface))
pipeline.add(SynixSearch("reference-search", surface=reference_surface))

# Flat file projections
pipeline.add(FlatFile("context-doc", sources=[core, work_status, user_model]))
pipeline.add(FlatFile("weekly-brief", sources=[short_window, open_threads]))
