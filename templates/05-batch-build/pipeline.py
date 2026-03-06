# pipeline.py — Batch Build Demo
#
# DAG:
#   Level 0: bios [Source]                -> one artifact per person
#   Level 1: work_styles [1:1, batch]     -> work style per person (OpenAI Batch API)
#   Level 2: team_summary [N:1, sync]     -> single team summary (Anthropic sync)
#
# Demonstrates:
#   - Batch-mode layer (OpenAI) at level 1, processed via Batch API
#   - Sync-mode layer (Anthropic) at level 2, normal inference
#   - Mixed providers in a single pipeline
#
# Usage:
#   uvx synix batch-build run pipeline.py --poll
#   uvx synix batch-build plan pipeline.py
#   uvx synix list

from synix import Pipeline, SearchSurface, Source, SynixSearch, Transform
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact

# -- Transforms --------------------------------------------------------------


class WorkStyleBatchTransform(Transform):
    """One bio -> one work style profile. Runs via OpenAI Batch API."""

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
                prompt_id="work_style_batch_v1",
                model_config=config.get("llm_config"),
            )
        ]


class TeamSummaryTransform(Transform):
    """Roll up all work styles into a single team summary. Runs synchronously."""

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
                        "team summary (3-5 sentences) covering how these styles complement "
                        "each other, natural collaboration patterns, and collective "
                        "strengths.\n\n"
                        f"{profiles}"
                    ),
                }
            ],
            artifact_desc="team-summary",
        )
        return [
            Artifact(
                label="team-summary",
                artifact_type="team_summary",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id="team_summary_v1",
                model_config=config.get("llm_config"),
            )
        ]


# -- Pipeline definition -----------------------------------------------------

pipeline = Pipeline("batch-build-demo")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Level 0 — bios
bios = Source("bios", dir="./sources/bios")

# Level 1 — per-person work style via OpenAI Batch API
work_styles = WorkStyleBatchTransform(
    "work_styles",
    depends_on=[bios],
    batch=True,
    config={
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 512,
        }
    },
)

# Level 2 — team summary via Anthropic (sync)
team_summary = TeamSummaryTransform(
    "team_summary",
    depends_on=[work_styles],
    batch=False,
)

team_search = SearchSurface(
    "team-search",
    sources=[bios, work_styles, team_summary],
    modes=["fulltext"],
)

pipeline.add(bios, work_styles, team_summary, team_search)

# SynixSearch — searchable local output across all layers
pipeline.add(
    SynixSearch("search", surface=team_search)
)
