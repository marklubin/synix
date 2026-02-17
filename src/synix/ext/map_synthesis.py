"""MapSynthesis — configurable 1:1 transform.

Applies a prompt template to each input artifact independently.
"""

from __future__ import annotations

import hashlib
import inspect
import logging

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform
from synix.ext._render import render_template

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
        prompt: str,
        label_fn: object | None = None,
        artifact_type: str = "summary",
        config: dict | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config, batch=batch)
        self.prompt = prompt
        self.label_fn = label_fn
        self.artifact_type = artifact_type

    def get_cache_key(self, config: dict) -> str:
        """Include prompt and artifact_type in cache key."""
        parts = f"{self.prompt}\x00{self.artifact_type}"
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint component if label_fn is set."""
        fp = super().compute_fingerprint(config)
        if self.label_fn is not None:
            from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

            components = dict(fp.components)
            try:
                components["callable"] = fingerprint_value(inspect.getsource(self.label_fn))
            except (OSError, TypeError):
                components["callable"] = fingerprint_value(repr(self.label_fn))
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        model_config = config.get("llm_config", {})
        prompt_id = self._make_prompt_id()

        inp = inputs[0]
        rendered = render_template(
            self.prompt,
            artifact=inp.content,
            label=inp.label,
            artifact_type=inp.artifact_type,
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": rendered}],
            artifact_desc=f"{self.name} {inp.label}",
        )

        if self.label_fn is not None:
            label = self.label_fn(inp)
        else:
            label = f"{self.name}-{inp.label}"

        return [
            Artifact(
                label=label,
                artifact_type=self.artifact_type,
                content=response.content,
                input_ids=[inp.artifact_id],
                prompt_id=prompt_id,
                model_config=model_config,
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"map_synthesis_v{hash_prefix}"
