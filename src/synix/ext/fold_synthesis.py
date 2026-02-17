"""FoldSynthesis — configurable N:1 sequential accumulation transform.

Processes inputs one at a time, building up an accumulated result through
sequential LLM calls.
"""

from __future__ import annotations

import hashlib
import inspect
import logging

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform
from synix.ext._render import render_template

logger = logging.getLogger(__name__)


class FoldSynthesis(Transform):
    """N:1 sequential transform — accumulate through inputs one at a time.

    Example::

        progressive = FoldSynthesis(
            "progressive-summary",
            depends_on=[episodes],
            prompt="Update this running summary:\\n\\nCurrent:\\n{accumulated}\\n\\nNew:\\n{artifact}",
            initial="No information yet.",
            label="progressive-summary",
            artifact_type="progressive_summary",
        )

    Placeholders: ``{accumulated}`` (running accumulator), ``{artifact}`` (current input content),
    ``{label}`` (current input label), ``{step}`` (1-based step number),
    ``{total}`` (total input count).
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list | None = None,
        prompt: str,
        initial: str = "",
        sort_by: str | object | None = None,
        label: str,
        artifact_type: str = "summary",
        config: dict | None = None,
    ):
        # FoldSynthesis is inherently sequential, never batch
        super().__init__(name, depends_on=depends_on, config=config, batch=False)
        self.prompt = prompt
        self.initial = initial
        self.sort_by = sort_by
        self.label_value = label
        self.artifact_type = artifact_type

    def get_cache_key(self, config: dict) -> str:
        """Include prompt, initial, sort_by, and artifact_type in cache key."""
        sort_by_str = ""
        if self.sort_by is not None:
            sort_by_str = self.sort_by if isinstance(self.sort_by, str) else repr(self.sort_by)
        combined = f"{self.prompt}\x00{self.initial}\x00{sort_by_str}\x00{self.artifact_type}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint component if sort_by is a callable."""
        fp = super().compute_fingerprint(config)
        if self.sort_by is not None and callable(self.sort_by) and not isinstance(self.sort_by, str):
            from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

            components = dict(fp.components)
            try:
                components["callable"] = fingerprint_value(inspect.getsource(self.sort_by))
            except (OSError, TypeError):
                components["callable"] = fingerprint_value(repr(self.sort_by))
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        """N:1 — single unit (sequential by nature)."""
        return [(inputs, {})]

    def estimate_output_count(self, input_count: int) -> int:
        return 1

    def _sort_inputs(self, inputs: list[Artifact]) -> list[Artifact]:
        """Sort inputs according to sort_by parameter."""
        if self.sort_by is None:
            return sorted(inputs, key=lambda a: a.artifact_id)
        elif isinstance(self.sort_by, str):
            return sorted(inputs, key=lambda a: a.metadata.get(self.sort_by, ""))
        else:
            return sorted(inputs, key=self.sort_by)

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        model_config = config.get("llm_config", {})
        prompt_id = self._make_prompt_id()

        sorted_inputs = self._sort_inputs(inputs)
        accumulated = self.initial
        total = len(sorted_inputs)

        for step, inp in enumerate(sorted_inputs, 1):
            rendered = render_template(
                self.prompt,
                accumulated=accumulated,
                artifact=inp.content,
                label=inp.label,
                step=str(step),
                total=str(total),
            )

            response = _logged_complete(
                client,
                config,
                messages=[{"role": "user", "content": rendered}],
                artifact_desc=f"{self.name} step {step}/{total}",
            )
            accumulated = response.content

        return [
            Artifact(
                label=self.label_value,
                artifact_type=self.artifact_type,
                content=accumulated,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"input_count": len(inputs)},
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"fold_synthesis_v{hash_prefix}"
