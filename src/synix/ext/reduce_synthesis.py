"""ReduceSynthesis — configurable N:1 transform.

Combines all input artifacts into a single output via a prompt template.
"""

from __future__ import annotations

import hashlib
import logging

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform
from synix.ext._render import render_template

logger = logging.getLogger(__name__)


class ReduceSynthesis(Transform):
    """N:1 transform — all inputs combined into a single output.

    Example::

        team_dynamics = ReduceSynthesis(
            "team_dynamics",
            depends_on=[work_styles],
            prompt="Analyze team dynamics from these profiles:\\n\\n{artifacts}",
            label="team-dynamics",
            artifact_type="team_dynamics",
        )

    Placeholders: ``{artifacts}`` (all input contents joined),
    ``{count}`` (input count).
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list | None = None,
        prompt: str,
        label: str,
        artifact_type: str = "summary",
        config: dict | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config, batch=batch)
        self.prompt = prompt
        self.label_value = label
        self.artifact_type = artifact_type

    def get_cache_key(self, config: dict) -> str:
        """Include prompt and artifact_type in cache key."""
        parts = f"{self.prompt}\x00{self.artifact_type}"
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        """N:1 — all inputs in a single unit."""
        return [(inputs, {})]

    def estimate_output_count(self, input_count: int) -> int:
        return 1

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        model_config = config.get("llm_config", {})
        prompt_id = self._make_prompt_id()

        # Sort inputs by artifact_id for deterministic prompt -> stable cassette key
        sorted_inputs = sorted(inputs, key=lambda a: a.artifact_id)
        artifacts_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in sorted_inputs)

        rendered = render_template(
            self.prompt,
            artifacts=artifacts_text,
            count=str(len(inputs)),
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": rendered}],
            artifact_desc=f"{self.name}",
        )

        return [
            Artifact(
                label=self.label_value,
                artifact_type=self.artifact_type,
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"input_count": len(inputs)},
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"reduce_synthesis_v{hash_prefix}"
