"""GroupSynthesis — configurable N:M group-by transform.

Groups input artifacts by a metadata key or callable, produces one output per group.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform, TransformContext
from synix.ext._render import render_template
from synix.ext._util import stable_callable_repr

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
        uses: list | None = None,
        group_by: str | Callable,
        prompt: str,
        label_prefix: str | None = None,
        metadata_fn: Callable | None = None,
        artifact_type: str = "summary",
        on_missing: str = "group",
        missing_key: str = "_ungrouped",
        config: dict | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, uses=uses, config=config, batch=batch)
        self.group_by = group_by
        self.prompt = prompt
        self.label_prefix = label_prefix
        self.metadata_fn = metadata_fn
        self.artifact_type = artifact_type
        if on_missing not in ("group", "skip", "error"):
            raise ValueError(f"on_missing must be 'group', 'skip', or 'error', got {on_missing!r}")
        self.on_missing = on_missing
        self.missing_key = missing_key

    def get_cache_key(self, config: dict) -> str:
        """Include prompt, group_by, on_missing, missing_key, artifact_type, and metadata_fn in cache key."""
        group_by_str = self.group_by if isinstance(self.group_by, str) else stable_callable_repr(self.group_by)
        metadata_fn_str = stable_callable_repr(self.metadata_fn) if self.metadata_fn is not None else ""
        parts = (
            f"{self.prompt}\x00{group_by_str}\x00{self.on_missing}"
            f"\x00{self.missing_key}\x00{self.artifact_type}\x00{metadata_fn_str}"
        )
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint components for group_by and metadata_fn."""
        fp = super().compute_fingerprint(config)
        callables = {}
        if callable(self.group_by) and not isinstance(self.group_by, str):
            callables["group_by"] = self.group_by
        if self.metadata_fn is not None:
            callables["metadata_fn"] = self.metadata_fn
        if callables:
            from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

            components = dict(fp.components)
            for key, fn in callables.items():
                try:
                    components[key] = fingerprint_value(inspect.getsource(fn))
                except (OSError, TypeError):
                    components[key] = fingerprint_value(repr(fn))
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def _get_group_key(self, artifact: Artifact) -> str | None:
        """Extract group key from an artifact. Returns None if missing."""
        if callable(self.group_by) and not isinstance(self.group_by, str):
            return self.group_by(artifact)
        return artifact.metadata.get(self.group_by)

    def split(self, inputs: list[Artifact], ctx: TransformContext) -> list[tuple[list[Artifact], dict]]:
        """Group by metadata key or callable, one unit per group."""
        ctx = self.get_context(ctx)
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

    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        ctx = self.get_context(ctx)
        group_key = ctx.get("_group_key")
        if group_key is None:
            # Called directly without split — process all groups sequentially
            results: list[Artifact] = []
            for unit_inputs, config_extras in self.split(inputs, ctx):
                results.extend(self.execute(unit_inputs, ctx.with_updates(config_extras)))
            return results

        client = _get_llm_client(ctx)
        model_config = ctx.llm_config
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
            ctx,
            messages=[{"role": "user", "content": rendered}],
            artifact_desc=f"{self.name} group-{group_key}",
        )

        prefix = self.label_prefix or (self.group_by if isinstance(self.group_by, str) else self.name)
        slug = group_key.lower().replace(" ", "-")
        label = f"{prefix}-{slug}"

        output_metadata = {"group_key": group_key, "input_count": len(inputs)}
        if self.metadata_fn is not None:
            output_metadata.update(self.metadata_fn(group_key, inputs))

        return [
            Artifact(
                label=label,
                artifact_type=self.artifact_type,
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata=output_metadata,
            )
        ]

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"group_synthesis_v{hash_prefix}"
