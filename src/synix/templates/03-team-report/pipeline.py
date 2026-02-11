# pipeline.py — My First Synix Pipeline
#
# DAG:
#   Level 0: bios [parse]              → one artifact per person
#   Level 0: project_brief [parse]     → the task description
#   Level 1: work_styles [work_style]  → inferred work style per person (1:1)
#   Level 2: team_dynamics [dynamics]  → rolled-up team dynamics analysis
#   Level 3: final_report [report]     → synthesis of team + project brief
#
# Usage:
#   synix build pipeline.py
#   synix validate pipeline.py
#   synix search 'hiking'

from synix import Layer, Pipeline, Projection, ValidatorDecl
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.build.transforms import BaseTransform, register_transform
from synix.build.validators import BaseValidator, Violation, register_validator
from synix.core.models import Artifact

# -- Transforms --------------------------------------------------------------

@register_transform("work_style")
class WorkStyleTransform(BaseTransform):
    """One bio → one work style profile. Default split gives 1:1."""

    def execute(self, inputs, config):
        bio = inputs[0]
        client = _get_llm_client(config)
        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": (
                "Given this person's background, infer their likely work style "
                "in 2-3 sentences. Cover how they approach problems, what role "
                "they naturally take in a team, and how their interests shape "
                "their work.\n\n"
                f"{bio.content}"
            )}],
            artifact_desc=f"work-style {bio.artifact_id}",
        )
        safe_id = bio.artifact_id.replace("t-text-", "")
        return [Artifact(
            artifact_id=f"ws-{safe_id}",
            artifact_type="work_style",
            content=response.content,
            input_hashes=[bio.content_hash],
            prompt_id="work_style_v1",
            model_config=config.get("llm_config"),
        )]


@register_transform("dynamics")
class TeamDynamicsTransform(BaseTransform):
    """Roll up all work style profiles into one team dynamics analysis."""

    def split(self, inputs, config):
        return [(inputs, {})]

    def execute(self, inputs, config):
        client = _get_llm_client(config)
        profiles = "\n\n".join(
            f"- {a.artifact_id}: {a.content}"
            for a in sorted(inputs, key=lambda a: a.artifact_id)
        )
        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": (
                "These are work style profiles for each team member. Write a "
                "team dynamics analysis (3-5 sentences) covering how these "
                "styles complement each other, natural collaboration patterns, "
                "potential friction points, and collective strengths.\n\n"
                f"{profiles}"
            )}],
            artifact_desc="team-dynamics",
        )
        return [Artifact(
            artifact_id="team-dynamics",
            artifact_type="team_dynamics",
            content=response.content,
            input_hashes=[a.content_hash for a in inputs],
            prompt_id="team_dynamics_v1",
            model_config=config.get("llm_config"),
        )]


@register_transform("report")
class FinalReportTransform(BaseTransform):
    """Combine team dynamics + project brief into a staffing report."""

    def split(self, inputs, config):
        return [(inputs, {})]

    def execute(self, inputs, config):
        client = _get_llm_client(config)
        dynamics = next(
            (a for a in inputs if a.artifact_type == "team_dynamics"), None
        )
        brief = next(
            (a for a in inputs if a.artifact_type == "transcript"), None
        )
        dynamics_text = dynamics.content if dynamics else "(no team dynamics)"
        brief_text = brief.content if brief else "(no project brief)"

        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": (
                "Given this team dynamics analysis and project brief, write a "
                "staffing report (4-6 sentences). Cover who should own which "
                "parts of the work, where the team's strengths align with "
                "project needs, risks or skill gaps, and how they should "
                "organize.\n\n"
                f"Team dynamics:\n{dynamics_text}\n\n"
                f"Project brief:\n{brief_text}"
            )}],
            artifact_desc="final-report",
        )
        return [Artifact(
            artifact_id="final-report",
            artifact_type="final_report",
            content=response.content,
            input_hashes=[a.content_hash for a in inputs],
            prompt_id="final_report_v1",
            model_config=config.get("llm_config"),
        )]


# -- Validator ----------------------------------------------------------------

@register_validator("max_length")
class MaxLengthValidator(BaseValidator):
    """Check that artifact content is under a maximum character count."""

    def validate(self, artifacts, ctx):
        config = getattr(self, "_config", {})
        max_chars = config.get("max_chars", 2000)
        violations = []
        for a in artifacts:
            if len(a.content) > max_chars:
                violations.append(Violation(
                    violation_type="max_length",
                    severity="error",
                    message=f"Content is {len(a.content)} chars (max {max_chars})",
                    artifact_id=a.artifact_id,
                    field="content",
                ))
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
pipeline.add_layer(Layer(
    name="bios",
    level=0,
    transform="parse",
    config={"source_dir": "./sources/bios"},
))

pipeline.add_layer(Layer(
    name="project_brief",
    level=0,
    transform="parse",
    config={"source_dir": "./sources/brief"},
))

# Level 1 — per-person work style (1:1)
pipeline.add_layer(Layer(
    name="work_styles",
    level=1,
    depends_on=["bios"],
    transform="work_style",
))

# Level 2 — team dynamics rollup (many:1)
pipeline.add_layer(Layer(
    name="team_dynamics",
    level=2,
    depends_on=["work_styles"],
    transform="dynamics",
))

# Level 3 — final report (team dynamics + project brief → synthesis)
pipeline.add_layer(Layer(
    name="final_report",
    level=3,
    depends_on=["team_dynamics", "project_brief"],
    transform="report",
))

# Projection — every layer searchable
pipeline.add_projection(Projection(
    name="search",
    projection_type="search_index",
    sources=[
        {"layer": "bios", "search": ["fulltext"]},
        {"layer": "project_brief", "search": ["fulltext"]},
        {"layer": "work_styles", "search": ["fulltext"]},
        {"layer": "team_dynamics", "search": ["fulltext"]},
        {"layer": "final_report", "search": ["fulltext"]},
    ],
))

# Validator — final report must be concise
pipeline.add_validator(ValidatorDecl(
    name="max_length",
    config={"layers": ["final_report"], "max_chars": 5000},
))
