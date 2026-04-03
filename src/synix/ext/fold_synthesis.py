"""FoldSynthesis — configurable N:1 sequential accumulation transform.

Processes inputs one at a time, building up an accumulated result through
sequential LLM calls. Supports incremental checkpoint resume: when new inputs
arrive and the transform config hasn't changed, only the delta is folded.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Callable

from synix.build.fingerprint import Fingerprint
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact, Transform, TransformContext
from synix.ext._render import render_template
from synix.ext._util import stable_callable_repr

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
        uses: list | None = None,
        prompt: str,
        initial: str = "",
        sort_by: str | Callable | None = None,
        label: str,
        metadata_fn: Callable | None = None,
        artifact_type: str = "summary",
        config: dict | None = None,
    ):
        # FoldSynthesis is inherently sequential, never batch
        super().__init__(name, depends_on=depends_on, uses=uses, config=config, batch=False)
        self.prompt = prompt
        self.initial = initial
        self.sort_by = sort_by
        self.label_value = label
        self.metadata_fn = metadata_fn
        self.artifact_type = artifact_type

    def get_cache_key(self, config: dict) -> str:
        """Include prompt, initial, sort_by, artifact_type, and metadata_fn in cache key."""
        sort_by_str = ""
        if self.sort_by is not None:
            sort_by_str = self.sort_by if isinstance(self.sort_by, str) else stable_callable_repr(self.sort_by)
        metadata_fn_str = stable_callable_repr(self.metadata_fn) if self.metadata_fn is not None else ""
        combined = f"{self.prompt}\x00{self.initial}\x00{sort_by_str}\x00{self.artifact_type}\x00{metadata_fn_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint components for sort_by and metadata_fn."""
        fp = super().compute_fingerprint(config)
        callables = {}
        if self.sort_by is not None and callable(self.sort_by) and not isinstance(self.sort_by, str):
            callables["sort_by"] = self.sort_by
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

    def split(self, inputs: list[Artifact], ctx: TransformContext) -> list[tuple[list[Artifact], dict]]:
        """N:1 — single unit (sequential by nature)."""
        ctx = self.get_context(ctx)
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

    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        ctx = self.get_context(ctx)
        client = _get_llm_client(ctx)
        model_config = ctx.llm_config
        prompt_id = self._make_prompt_id()

        sorted_inputs = self._sort_inputs(inputs)
        transform_fp = self.compute_fingerprint(
            ctx.to_dict() if hasattr(ctx, "to_dict") else ctx
        )

        # --- Checkpoint resume logic ---
        previous = ctx.get("_previous_artifact") if hasattr(ctx, "get") else None
        resume = self._try_resume(previous, sorted_inputs, transform_fp)

        if resume is not None:
            new_inputs, accumulated, start_step = resume
            if new_inputs is None:
                # No new inputs — return previous artifact unchanged
                return [previous]
        else:
            # No valid checkpoint — full compute
            new_inputs = sorted_inputs
            accumulated = self.initial
            start_step = 0

        total = len(sorted_inputs)

        # --- Main fold loop (over new_inputs only) ---
        for step, inp in enumerate(new_inputs, start_step + 1):
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
                ctx,
                messages=[{"role": "user", "content": rendered}],
                artifact_desc=f"{self.name} step {step}/{total}",
            )
            accumulated = response.content

        # --- Persist checkpoint ---
        all_input_labels = [a.label for a in sorted_inputs]
        output_metadata = {"input_count": len(inputs)}
        if self.metadata_fn is not None:
            output_metadata.update(self.metadata_fn(inputs))
        output_metadata["_fold_checkpoint"] = {
            "accumulated": accumulated,
            "seen_input_labels": all_input_labels,
            "transform_fingerprint": transform_fp.to_dict(),
        }

        return [
            Artifact(
                label=self.label_value,
                artifact_type=self.artifact_type,
                content=accumulated,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata=output_metadata,
            )
        ]

    def _try_resume(
        self,
        previous: Artifact | None,
        sorted_inputs: list[Artifact],
        transform_fp: Fingerprint,
    ) -> tuple[list[Artifact] | None, str, int] | None:
        """Attempt checkpoint resume.

        Returns ``(new_inputs, accumulated, start_step)`` or ``None`` if a full
        recompute is needed.  ``new_inputs=None`` means nothing changed.
        """
        if previous is None:
            return None

        checkpoint = previous.metadata.get("_fold_checkpoint")
        if checkpoint is None:
            return None

        # 1. Transform fingerprint must match
        stored_fp = Fingerprint.from_dict(checkpoint.get("transform_fingerprint"))
        if stored_fp is None or not transform_fp.matches(stored_fp):
            logger.info("%s: transform changed, full recompute", self.name)
            return None

        # 2. Checkpoint integrity: accumulated must equal artifact content
        if checkpoint.get("accumulated") != previous.content:
            logger.warning(
                "%s: checkpoint inconsistent with artifact content, full recompute",
                self.name,
            )
            return None

        # 3. All previously-seen inputs must still be present
        seen_labels = checkpoint.get("seen_input_labels", [])
        seen_set = set(seen_labels)
        current_set = set(a.label for a in sorted_inputs)

        if not seen_set.issubset(current_set):
            removed = seen_set - current_set
            logger.info(
                "%s: %d inputs removed (%s...), full recompute",
                self.name,
                len(removed),
                list(removed)[:3],
            )
            return None

        # 4. Identify new inputs
        new_inputs = [a for a in sorted_inputs if a.label not in seen_set]

        if not new_inputs:
            return (None, previous.content, len(seen_labels))  # no change

        # 5. New inputs must all sort after seen inputs (no interleave)
        last_seen_idx = max(
            (i for i, a in enumerate(sorted_inputs) if a.label in seen_set),
            default=-1,
        )
        first_new_idx = min(
            i for i, a in enumerate(sorted_inputs) if a.label not in seen_set
        )
        if first_new_idx <= last_seen_idx:
            logger.info(
                "%s: new inputs interleave with seen inputs, full recompute",
                self.name,
            )
            return None

        # All checks passed — safe to resume
        logger.info(
            "%s: resuming from checkpoint, %d new inputs (of %d total)",
            self.name,
            len(new_inputs),
            len(sorted_inputs),
        )
        return (new_inputs, checkpoint["accumulated"], len(seen_labels))

    def _make_prompt_id(self) -> str:
        hash_prefix = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        return f"fold_synthesis_v{hash_prefix}"
