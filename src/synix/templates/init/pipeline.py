# pipeline.py — My First Synix Pipeline
#
# This pipeline reads markdown bios from sources/, combines them into
# a team profile using an LLM, validates the output length, and makes
# everything searchable.
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

# -- Custom transform: combine all bios into one team profile ----------------

@register_transform("team_profile")
class TeamProfileTransform(BaseTransform):
    """Combine individual bios into a single team profile summary."""

    def split(self, inputs, config):
        return [(inputs, {})]

    def execute(self, inputs, config):
        client = _get_llm_client(config)
        bios = "\n\n".join(a.content for a in sorted(inputs, key=lambda a: a.artifact_id))
        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": (
                "Write a short team profile (3-5 sentences) summarizing these people. "
                "Mention each person's role, location, and one personal detail.\n\n"
                f"{bios}"
            )}],
            artifact_desc="team-profile",
        )
        return [Artifact(
            artifact_id="team-profile",
            artifact_type="team_profile",
            content=response.content,
            input_hashes=[a.content_hash for a in inputs],
            prompt_id="team_profile_v1",
            model_config=config.get("llm_config"),
        )]


# -- Custom validator: check content length ----------------------------------

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

pipeline.add_layer(Layer(
    name="bios",
    level=0,
    transform="parse",
))

pipeline.add_layer(Layer(
    name="team_profile",
    level=1,
    depends_on=["bios"],
    transform="team_profile",
))

pipeline.add_projection(Projection(
    name="search",
    projection_type="search_index",
    sources=[
        {"layer": "bios", "search": ["fulltext"]},
        {"layer": "team_profile", "search": ["fulltext"]},
    ],
))

pipeline.add_validator(ValidatorDecl(
    name="max_length",
    config={"layers": ["team_profile"], "max_chars": 2000},
))
