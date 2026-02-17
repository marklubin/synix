"""GroupSynthesis — configurable N:M group-by transform.

Groups input artifacts by a metadata key or callable, produces one output per group.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections import defaultdict

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform
from synix.ext._render import render_template

logger = logging.getLogger(__name__)


class GroupSynthesis(Transform):
    """N:M group-by transform — group inputs, one output per group.

    Example::

        by_customer = GroupSynthesis(
            "customer-summaries",
            depends_on=[episodes],
            group_by="customer_id",
            prompt="Summarize interactions for customer '{group_key}':\\n\\n{artifacts}",
            artifact_type="customer_summary",
        )

    Placeholders: ``{group_key}``, ``{artifacts}``, ``{count}``, ``{artifact_type}``.

    ``on_missing`` controls behavior when the group_by key is absent:
      - ``"group"``: collect under ``missing_key`` (default ``"_ungrouped"``), warn
      - ``"skip"``: drop artifacts without the key, warn
      - ``"error"``: raise ``ValueError`` immediately
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list | None = None,
        group_by: str | object,
        prompt: str,
        label_prefix: str | None = None,
        artifact_type: str = "summary",
        on_missing: str = "group",
        missing_key: str = "_ungrouped",
        config: dict | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config, batch=batch)
        self.group_by = group_by
        self.prompt = prompt
        self.label_prefix = label_prefix
        self.artifact_type = artifact_type
        if on_missing not in ("group", "skip", "error"):
            raise ValueError(f"on_missing must be 'group', 'skip', or 'error', got {on_missing!r}")
        self.on_missing = on_missing
        self.missing_key = missing_key

    def get_cache_key(self, config: dict) -> str:
        """Include prompt, group_by, on_missing, missing_key, and artifact_type in cache key."""
        group_by_str = self.group_by if isinstance(self.group_by, str) else repr(self.group_by)
        parts = f"{self.prompt}\x00{group_by_str}\x00{self.on_missing}\x00{self.missing_key}\x00{self.artifact_type}"
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint component if group_by is a callable."""
        fp = super().compute_fingerprint(config)
        if callable(self.group_by) and not isinstance(self.group_by, str):
            from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

            components = dict(fp.components)
            try:
                components["callable"] = fingerprint_value(inspect.getsource(self.group_by))
            except (OSError, TypeError):
                components["callable"] = fingerprint_value(repr(self.group_by))
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def _get_group_key(self, artifact: Artifact) -> str | None:
        """Extract group key from an artifact. Returns None if missing."""
        if callable(self.group_by) and not isinstance(self.group_by, str):
            return self.group_by(artifact)
        return artifact.metadata.get(self.group_by)

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        """Group by metadata key or callable, one unit per group."""
        groups: dict[str, list[Artifact]] = defaultdict(list)
        missing_count = 0

        for artifact in inputs:
            key = self._get_group_key(artifact)
            if key is None:
                missing_count += 1
                if self.on_missing == "error":
                    field_desc = self.group_by if isinstance(self.group_by, str) else "callable"
                    raise ValueError(
                        f"GroupSynthesis '{self.name}': artifact '{artifact.label}' missing field '{field_desc}'"
                    )
                elif self.on_missing == "group":
                    groups[self.missing_key].append(artifact)
                # on_missing == "skip": just drop it
            else:
                groups[key].append(artifact)

        if missing_count > 0:
            field_desc = self.group_by if isinstance(self.group_by, str) else "callable"
            if self.on_missing == "group":
                logger.warning(
                    "GroupSynthesis '%s': %d artifact(s) missing field '%s', grouped as '%s'",
                    self.name,
                    missing_count,
                    field_desc,
                    self.missing_key,
                )
            elif self.on_missing == "skip":
                logger.warning(
                    "GroupSynthesis '%s': %d artifact(s) missing field '%s', skipped",
                    self.name,
                    missing_count,
                    field_desc,
                )

        return [(artifacts, {"_group_key": group_key}) for group_key, artifacts in sorted(groups.items())]

    def estimate_output_count(self, input_count: int) -> int:
        return max(input_count // 2, 1)

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        group_key = config.get("_group_key")
        if group_key is None:
            # Called directly without split — process all groups sequentially
            results: list[Artifact] = []
            for unit_inputs, config_extras in self.split(inputs, config):
                merged = {**config, **config_extras}
                results.extend(self.execute(unit_inputs, merged))
            return results

        client = _get_llm_client(config)
        model_config = config.get("llm_config", {})
        prompt_id = self._make_prompt_id()

        # Sort inputs by artifact_id for deterministic prompt -> stable cassette key
        sorted_inputs = sorted(inputs, key=lambda a: a.artifact_id)
        artifacts_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in sorted_inputs)

        rendered = render_template(
            self.prompt,
            group_key=group_key,
            artifacts=artifacts_text,
            count=str(len(inputs)),
            artifact_type=self.artifact_type,
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": rendered}],
            artifact_desc=f"{self.name} group-{group_key}",
        )

        prefix = self.label_prefix or (self.group_by if isinstance(self.group_by, str) else self.name)
        slug = group_key.lower().replace(" ", "-")
        label = f"{prefix}-{slug}"

        return [
            Artifact(
                label=label,
                artifact_type=self.artifact_type,
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"group_key": group_key, "input_count": len(inputs)},
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"group_synthesis_v{hash_prefix}"
