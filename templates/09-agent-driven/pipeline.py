# pipeline.py — Agent-Driven Team Report
#
# Same DAG as template 03-team-report, but using named agents instead
# of inline prompts. Agents define persona/semantics (HOW), transforms
# define task structure (WHAT).
#
# DAG:
#   Level 0: bios [source]         → one artifact per person
#   Level 0: brief [source]        → the project description
#   Level 1: work_styles [map]     → inferred work style per person (agent: analyst)
#   Level 2: team_dynamics [reduce] → team dynamics analysis (agent: analyst)
#   Level 3: final_report [fold]   → staffing report (agent: reporter)
#
# Agents:
#   analyst  — concise analytical writer (map + reduce)
#   reporter — structured report writer (fold)
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix release HEAD --to local
#   uvx synix search "collaboration"

# -- Agent setup --------------------------------------------------------------
# Create a PromptStore and seed instructions from files, then bind to agents.
# In production with `synix serve`, the server does this automatically.
from pathlib import Path

from synix import Pipeline, SearchSurface, Source, SynixSearch
from synix.agents import SynixLLMAgent
from synix.server.prompt_store import PromptStore
from synix.transforms import FoldSynthesis, MapSynthesis, ReduceSynthesis

_here = Path(__file__).parent
_store = PromptStore(_here / ".synix" / "prompts.db")

# Seed instructions from files (no-op if already seeded)
_store.seed_from_files(_here / "prompts")

analyst = SynixLLMAgent(
    name="analyst",
    prompt_key="analyst",
    description="Concise analytical writer — infers patterns and dynamics",
)
analyst.bind_prompt_store(_store)

reporter = SynixLLMAgent(
    name="reporter",
    prompt_key="reporter",
    description="Structured report writer — builds documents incrementally",
)
reporter.bind_prompt_store(_store)

# -- Pipeline definition ------------------------------------------------------

pipeline = Pipeline("agent-driven-report")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Level 0 — sources
bios = Source("bios", dir="./sources/bios")
brief = Source("brief", dir="./sources/brief")

# Level 1 — per-person work style (analyst agent)
# Transform prompt defines the TASK, agent defines the PERSONA
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt=(
        "Given this person's background, infer their likely work style. "
        "Cover how they approach problems, what role they take in a team, "
        "and how their interests shape their work.\n\n"
        "{artifact}"
    ),
    agent=analyst,
    label_fn=lambda a: f"ws-{a.label.replace('t-text-', '')}",
    artifact_type="work_style",
)

# Level 2 — team dynamics (same analyst agent, different task)
team_dynamics = ReduceSynthesis(
    "team_dynamics",
    depends_on=[work_styles],
    prompt=(
        "These are work style profiles for each team member. Write a "
        "team dynamics analysis covering complementary styles, natural "
        "collaboration patterns, friction points, and strengths.\n\n"
        "{artifacts}"
    ),
    agent=analyst,
    label="team-dynamics",
    artifact_type="team_dynamics",
)

# Level 3 — final report (reporter agent, fold)
final_report = FoldSynthesis(
    "final_report",
    depends_on=[team_dynamics, brief],
    prompt=(
        "Update the staffing report with the new information.\n\n"
        "Current draft:\n{accumulated}\n\n"
        "New input:\n{artifact}"
    ),
    agent=reporter,
    initial="No report yet.",
    label="final-report",
    artifact_type="final_report",
)

# Search surface
report_search = SearchSurface(
    "report-search",
    sources=[bios, brief, work_styles, team_dynamics, final_report],
    modes=["fulltext"],
)

pipeline.add(bios, brief, work_styles, team_dynamics, final_report, report_search)
pipeline.add(SynixSearch("search", surface=report_search))
