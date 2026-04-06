"""MapSynthesis — configurable 1:1 transform.

Applies a prompt template to each input artifact independently.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform, TransformContext
from synix.ext._render import render_template
from synix.ext._util import stable_callable_repr

if TYPE_CHECKING:
    from synix.agents import Agent

logger = logging.getLogger(__name__)


class MapSynthesis(Transform):
    """1:1 transform — apply a prompt to each input independently.

    Example::

        work_styles = MapSynthesis(
            "work_styles",
            depends_on=[bios],
            prompt="Given this person's background, infer their work style:\\n\\n{artifact}",
            artifact_type="work_style",
        )

    Placeholders: ``{artifact}`` (input content), ``{label}`` (input label),
    ``{artifact_type}`` (input type).
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list | None = None,
        uses: list | None = None,
        prompt: str,
        label_fn: Callable | None = None,
        metadata_fn: Callable | None = None,
        artifact_type: str = "summary",
        agent: Agent | None = None,
        config: dict | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, uses=uses, config=config, batch=batch)
        self.prompt = prompt
        self.label_fn = label_fn
        self.metadata_fn = metadata_fn
        self.artifact_type = artifact_type
        self.agent = agent

    def get_cache_key(self, config: dict) -> str:
        """Include prompt, artifact_type, metadata_fn, and agent fingerprint in cache key."""
        metadata_fn_str = stable_callable_repr(self.metadata_fn) if self.metadata_fn is not None else ""
        agent_str = self.agent.fingerprint_value() if self.agent is not None else ""
        parts = f"{self.prompt}\x00{self.artifact_type}\x00{metadata_fn_str}\x00{agent_str}"
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint components for label_fn, metadata_fn, and agent."""
        fp = super().compute_fingerprint(config)
        from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

        components = dict(fp.components)
        modified = False

        callables = {}
        if self.label_fn is not None:
            callables["label_fn"] = self.label_fn
        if self.metadata_fn is not None:
            callables["metadata_fn"] = self.metadata_fn
        for key, fn in callables.items():
            modified = True
            try:
                components[key] = fingerprint_value(inspect.getsource(fn))
            except (OSError, TypeError):
                components[key] = fingerprint_value(repr(fn))

        if self.agent is not None:
            modified = True
            components["agent"] = self.agent.fingerprint_value()
            components.pop("model", None)

        if modified:
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        ctx = self.get_context(ctx)
        prompt_id = self._make_prompt_id()

        inp = inputs[0]
        rendered = render_template(
            self.prompt,
            artifact=inp.content,
            label=inp.label,
            artifact_type=inp.artifact_type,
        )

        if self.agent is not None:
            from synix.agents import AgentRequest

            result = self.agent.write(AgentRequest(
                prompt=rendered,
                metadata={
                    "transform_name": self.name,
                    "shape": "map",
                    "input_labels": [inp.label],
                    "artifact_type": self.artifact_type,
                },
            ))
            content = result.content
            model_config = None
            agent_fingerprint = self.agent.fingerprint_value()
        else:
            client = _get_llm_client(ctx)
            response = _logged_complete(
                client,
                ctx,
                messages=[{"role": "user", "content": rendered}],
                artifact_desc=f"{self.name} {inp.label}",
            )
            content = response.content
            model_config = ctx.llm_config
            agent_fingerprint = None

        if self.label_fn is not None:
            label = self.label_fn(inp)
        else:
            label = f"{self.name}-{inp.label}"

        # Propagate input metadata, overlaid with transform-specific fields
        output_metadata = dict(inp.metadata)
        if self.metadata_fn is not None:
            output_metadata.update(self.metadata_fn(inp))
        output_metadata["source_label"] = inp.label

        return [
            Artifact(
                label=label,
                artifact_type=self.artifact_type,
                content=content,
                input_ids=[inp.artifact_id],
                prompt_id=prompt_id,
                model_config=model_config,
                agent_fingerprint=agent_fingerprint,
                metadata=output_metadata,
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"map_synthesis_v{hash_prefix}"
