# pipeline.py — Team Report Pipeline (using synix.ext transforms)
#
# DAG:
#   Level 0: bios [parse]              -> one artifact per person
#   Level 0: project_brief [parse]     -> the task description
#   Level 1: work_styles [map]         -> inferred work style per person (1:1)
#   Level 2: team_dynamics [reduce]    -> rolled-up team dynamics analysis (N:1)
#   Level 3: final_report [fold]       -> staffing report built up sequentially
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix validate pipeline.py
#   uvx synix search 'hiking'

from synix import Pipeline, SearchIndex, Source
from synix.ext import FoldSynthesis, MapSynthesis, ReduceSynthesis
from synix.validators import RequiredField

# -- Pipeline definition -----------------------------------------------------

pipeline = Pipeline("my-first-pipeline")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Level 0 — two independent leaf nodes
bios = Source("bios", dir="./sources/bios")
project_brief = Source("project_brief", dir="./sources/brief")

# Level 1 — per-person work style (1:1 via MapSynthesis)
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt=(
        "Given this person's background, infer their likely work style "
        "in 2-3 sentences. Cover how they approach problems, what role "
        "they naturally take in a team, and how their interests shape "
        "their work.\n\n"
        "{artifact}"
    ),
    label_fn=lambda a: f"ws-{a.label.replace('t-text-', '')}",
    artifact_type="work_style",
)

# Level 2 — team dynamics rollup (N:1 via ReduceSynthesis)
team_dynamics = ReduceSynthesis(
    "team_dynamics",
    depends_on=[work_styles],
    prompt=(
        "These are work style profiles for each team member. Write a "
        "team dynamics analysis (3-5 sentences) covering how these "
        "styles complement each other, natural collaboration patterns, "
        "potential friction points, and collective strengths.\n\n"
        "{artifacts}"
    ),
    label="team-dynamics",
    artifact_type="team_dynamics",
)

# Level 3 — final report (fold through team dynamics + project brief)
final_report = FoldSynthesis(
    "final_report",
    depends_on=[team_dynamics, project_brief],
    prompt=(
        "You are writing a staffing report. Update it with the new "
        "information below.\n\n"
        "Current draft:\n{accumulated}\n\n"
        "New input:\n{artifact}"
    ),
    initial="No report yet.",
    label="final-report",
    artifact_type="final_report",
)

pipeline.add(bios, project_brief, work_styles, team_dynamics, final_report)

# Projection — every layer searchable
pipeline.add(
    SearchIndex(
        "search",
        sources=[bios, project_brief, work_styles, team_dynamics, final_report],
        search=["fulltext"],
    )
)

# Validator — final report must have input_count metadata
pipeline.add_validator(RequiredField(layers=[final_report], field="input_count"))
