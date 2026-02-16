# pipeline.py — My First Synix Pipeline
#
# DAG:
#   Level 0: bios [parse]              -> one artifact per person
#   Level 0: project_brief [parse]     -> the task description
#   Level 1: work_styles [work_style]  -> inferred work style per person (1:1)
#   Level 2: team_dynamics [dynamics]  -> rolled-up team dynamics analysis
#   Level 3: final_report [report]     -> synthesis of team + project brief
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix validate pipeline.py
#   uvx synix search 'hiking'

from synix import Pipeline, SearchIndex, Source, Transform
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact
from synix.validators import BaseValidator, Violation

# -- Transforms --------------------------------------------------------------


class WorkStyleTransform(Transform):
    """One bio -> one work style profile. Default split gives 1:1."""

    def execute(self, inputs, config):
        bio = inputs[0]
        client = _get_llm_client(config)
        response = _logged_complete(
            client,
            config,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Given this person's background, infer their likely work style "
                        "in 2-3 sentences. Cover how they approach problems, what role "
                        "they naturally take in a team, and how their interests shape "
                        "their work.\n\n"
                        f"{bio.content}"
                    ),
                }
            ],
            artifact_desc=f"work-style {bio.label}",
        )
        safe_id = bio.label.replace("t-text-", "")
        return [
            Artifact(
                label=f"ws-{safe_id}",
                artifact_type="work_style",
                content=response.content,
                input_ids=[bio.artifact_id],
                prompt_id="work_style_v1",
                model_config=config.get("llm_config"),
            )
        ]


class TeamDynamicsTransform(Transform):
    """Roll up all work style profiles into one team dynamics analysis."""

    def split(self, inputs, config):
        return [(inputs, {})]

    def execute(self, inputs, config):
        client = _get_llm_client(config)
        profiles = "\n\n".join(f"- {a.label}: {a.content}" for a in sorted(inputs, key=lambda a: a.label))
        response = _logged_complete(
            client,
            config,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "These are work style profiles for each team member. Write a "
                        "team dynamics analysis (3-5 sentences) covering how these "
                        "styles complement each other, natural collaboration patterns, "
                        "potential friction points, and collective strengths.\n\n"
                        f"{profiles}"
                    ),
                }
            ],
            artifact_desc="team-dynamics",
        )
        return [
            Artifact(
                label="team-dynamics",
                artifact_type="team_dynamics",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id="team_dynamics_v1",
                model_config=config.get("llm_config"),
            )
        ]


class FinalReportTransform(Transform):
    """Combine team dynamics + project brief into a staffing report."""

    def split(self, inputs, config):
        return [(inputs, {})]

    def execute(self, inputs, config):
        client = _get_llm_client(config)
        dynamics = next((a for a in inputs if a.artifact_type == "team_dynamics"), None)
        brief = next((a for a in inputs if a.artifact_type == "transcript"), None)
        dynamics_text = dynamics.content if dynamics else "(no team dynamics)"
        brief_text = brief.content if brief else "(no project brief)"

        response = _logged_complete(
            client,
            config,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Given this team dynamics analysis and project brief, write a "
                        "staffing report (4-6 sentences). Cover who should own which "
                        "parts of the work, where the team's strengths align with "
                        "project needs, risks or skill gaps, and how they should "
                        "organize.\n\n"
                        f"Team dynamics:\n{dynamics_text}\n\n"
                        f"Project brief:\n{brief_text}"
                    ),
                }
            ],
            artifact_desc="final-report",
        )
        return [
            Artifact(
                label="final-report",
                artifact_type="final_report",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id="final_report_v1",
                model_config=config.get("llm_config"),
            )
        ]


# -- Validator ----------------------------------------------------------------


class MaxLengthValidator(BaseValidator):
    """Check that artifact content is under a maximum character count."""

    name = "max_length"

    def __init__(self, *, layers: list, max_chars: int = 2000):
        self._layers = layers
        self._max_chars = max_chars

    def to_config_dict(self) -> dict:
        return {
            "layers": [l.name for l in self._layers],
            "max_chars": self._max_chars,
        }

    def validate(self, artifacts, ctx):
        violations = []
        for a in artifacts:
            if len(a.content) > self._max_chars:
                violations.append(
                    Violation(
                        violation_type="max_length",
                        severity="error",
                        message=f"Content is {len(a.content)} chars (max {self._max_chars})",
                        label=a.label,
                        field="content",
                    )
                )
        return violations


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

# Level 1 — per-person work style (1:1)
work_styles = WorkStyleTransform("work_styles", depends_on=[bios])

# Level 2 — team dynamics rollup (many:1)
team_dynamics = TeamDynamicsTransform("team_dynamics", depends_on=[work_styles])

# Level 3 — final report (team dynamics + project brief -> synthesis)
final_report = FinalReportTransform("final_report", depends_on=[team_dynamics, project_brief])

pipeline.add(bios, project_brief, work_styles, team_dynamics, final_report)

# Projection — every layer searchable
pipeline.add(
    SearchIndex(
        "search",
        sources=[bios, project_brief, work_styles, team_dynamics, final_report],
        search=["fulltext"],
    )
)

# Validator — final report must be concise
pipeline.add_validator(MaxLengthValidator(layers=[final_report], max_chars=5000))
