"""Fixer framework for automatically resolving validation violations.

Provides an ABC + decorator registry pattern (matching validators.py) for
fixers that propose and apply corrections to artifacts with full provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import ValidationResult, Violation, _store_llm_trace
from synix.core.models import Pipeline

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FixAction:
    """A proposed fix for a violation."""

    artifact_id: str
    action: str  # "rewrite", "redact", "skip", "unresolved"
    original_content_hash: str
    new_content: str  # proposed new content (empty for unresolved)
    new_content_hash: str
    description: str
    downstream_invalidated: list[str] = field(default_factory=list)
    evidence_source_ids: list[str] = field(default_factory=list)
    interactive: bool = False
    llm_explanation: str = ""


@dataclass
class FixResult:
    """Aggregated result of running fixers."""

    actions: list[FixAction] = field(default_factory=list)
    fixers_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rebuild_required: list[str] = field(default_factory=list)

    @property
    def fixed_count(self) -> int:
        return sum(1 for a in self.actions if a.action in ("rewrite", "redact"))

    @property
    def skipped_count(self) -> int:
        return sum(1 for a in self.actions if a.action in ("skip", "unresolved"))


@dataclass
class FixContext:
    """Context passed to fixers."""

    store: ArtifactStore
    provenance: ProvenanceTracker
    pipeline: Pipeline
    search_index: object | None = None  # SearchIndex when available
    llm_client: object | None = None  # LLMClient when available


# ---------------------------------------------------------------------------
# BaseFixer ABC + registry
# ---------------------------------------------------------------------------


class BaseFixer(ABC):
    """Abstract base class for fixers."""

    handles_violation_types: list[str] = []
    interactive: bool = False

    @abstractmethod
    def fix(self, violation: Violation, ctx: FixContext) -> FixAction:
        """Propose a fix for the given violation."""
        ...

    def can_handle(self, violation: Violation) -> bool:
        return violation.violation_type in self.handles_violation_types


_FIXERS: dict[str, type[BaseFixer]] = {}


def register_fixer(name: str):
    """Decorator to register a fixer class by name."""

    def wrapper(cls: type[BaseFixer]) -> type[BaseFixer]:
        _FIXERS[name] = cls
        return cls

    return wrapper


def get_fixer(name: str) -> BaseFixer:
    """Get an instantiated fixer by name."""
    if name not in _FIXERS:
        raise ValueError(f"Unknown fixer: {name}. Available: {list(_FIXERS.keys())}")
    return _FIXERS[name]()


# ---------------------------------------------------------------------------
# Downstream detection
# ---------------------------------------------------------------------------


def _find_downstream_artifacts(
    artifact_id: str,
    provenance: ProvenanceTracker,
) -> list[str]:
    """Find all artifacts that have artifact_id as a parent (direct children)."""
    downstream: list[str] = []
    for aid, rec in provenance._records.items():
        if artifact_id in rec.get("parent_artifact_ids", []):
            downstream.append(aid)
    return downstream


# ---------------------------------------------------------------------------
# Apply fix (after human approval)
# ---------------------------------------------------------------------------


def apply_fix(
    action: FixAction,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
) -> None:
    """Apply an approved fix: rewrite artifact content and update provenance.

    - Loads the existing artifact
    - Updates content and content_hash
    - Saves via store.save_artifact() (atomic write)
    - Records new provenance with evidence_source_ids as parents
    """
    artifact = store.load_artifact(action.artifact_id)
    if artifact is None:
        return

    # Get layer info from manifest
    manifest_entry = store._manifest.get(action.artifact_id, {})
    layer_name = manifest_entry.get("layer", artifact.metadata.get("layer_name", "unknown"))
    layer_level = manifest_entry.get("level", artifact.metadata.get("layer_level", 0))

    # Update artifact content
    artifact.content = action.new_content
    artifact.content_hash = f"sha256:{hashlib.sha256(action.new_content.encode()).hexdigest()}"

    # Save updated artifact
    store.save_artifact(artifact, layer_name, layer_level)

    # Re-record provenance with existing parents (evidence sources are
    # reference context for the fixer, not true input lineage).
    existing_parents = provenance.get_parents(action.artifact_id)
    provenance.record(
        action.artifact_id,
        parent_ids=existing_parents,
        prompt_id=artifact.prompt_id,
        model_config=artifact.model_config,
    )


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def run_fixers(
    validation_result: ValidationResult,
    pipeline: Pipeline,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
    search_index: object | None = None,
    llm_client: object | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> FixResult:
    """Run all fixers declared in the pipeline against violations.

    For each FixerDecl in pipeline.fixers:
    1. Instantiate the registered fixer
    2. Filter violations that this fixer can handle
    3. Call fix() for each matching violation
    4. Collect actions and compute downstream invalidation

    on_progress(message, current, total) is called before each fix attempt
    to allow the CLI to show live progress.

    Returns FixResult with proposed actions (not yet applied).
    """
    ctx = FixContext(store, provenance, pipeline, search_index, llm_client)
    result = FixResult()

    # Count total fixable violations across all fixers
    total_fixable = 0
    for decl in pipeline.fixers:
        fixer = get_fixer(decl.name)
        matching = [v for v in validation_result.violations if fixer.can_handle(v)]
        total_fixable += len(matching)

    current = 0
    for decl in pipeline.fixers:
        fixer = get_fixer(decl.name)
        fixer._config = decl.config  # type: ignore[attr-defined]

        matching = [v for v in validation_result.violations if fixer.can_handle(v)]

        for violation in matching:
            current += 1
            if on_progress:
                on_progress(
                    f"Fixing {violation.artifact_id} ({decl.name})",
                    current,
                    total_fixable,
                )
            try:
                action = fixer.fix(violation, ctx)
                # Compute downstream invalidation
                action.downstream_invalidated = _find_downstream_artifacts(action.artifact_id, provenance)
                result.actions.append(action)
                result.rebuild_required.extend(action.downstream_invalidated)
            except Exception as exc:
                result.errors.append(f"Fixer {decl.name} error on {violation.artifact_id}: {exc}")

        result.fixers_run.append(decl.name)

    # Deduplicate rebuild_required
    result.rebuild_required = list(set(result.rebuild_required))
    return result


# ---------------------------------------------------------------------------
# Built-in fixers
# ---------------------------------------------------------------------------


@register_fixer("semantic_enrichment")
class SemanticEnrichmentFixer(BaseFixer):
    """LLM-based fixer that resolves semantic conflicts using source context.

    Config:
        max_context_episodes: max source episodes to include (default 5)
        temperature: LLM temperature (default 0.3)
    """

    handles_violation_types = ["semantic_conflict"]
    interactive = True

    def fix(self, violation: Violation, ctx: FixContext) -> FixAction:
        artifact = ctx.store.load_artifact(violation.artifact_id)
        if artifact is None:
            return FixAction(
                artifact_id=violation.artifact_id,
                action="skip",
                original_content_hash="",
                new_content="",
                new_content_hash="",
                description="Artifact not found",
            )

        if ctx.llm_client is None:
            return FixAction(
                artifact_id=violation.artifact_id,
                action="skip",
                original_content_hash=artifact.content_hash,
                new_content="",
                new_content_hash="",
                description="No LLM client available",
            )

        config = getattr(self, "_config", {})
        max_context = config.get("max_context_episodes", 5)
        temperature = config.get("temperature", 0.3)

        claim_a = violation.metadata.get("claim_a", "")
        claim_b = violation.metadata.get("claim_b", "")
        claim_a_hint = violation.metadata.get("claim_a_source_hint", "")
        claim_b_hint = violation.metadata.get("claim_b_source_hint", "")

        # Search for source context
        evidence_source_ids: list[str] = []
        source_texts: list[str] = []

        if ctx.search_index is not None:
            search_queries = [claim_a, claim_b]
            if claim_a_hint:
                search_queries.append(claim_a_hint)
            if claim_b_hint:
                search_queries.append(claim_b_hint)

            seen_ids: set[str] = set()
            for query in search_queries:
                if not query.strip():
                    continue
                try:
                    results = ctx.search_index.query(query)
                    for r in results[:max_context]:
                        if r.artifact_id not in seen_ids:
                            seen_ids.add(r.artifact_id)
                            evidence_source_ids.append(r.artifact_id)
                            source_texts.append(f"[{r.artifact_id}]: {r.content[:500]}")
                except Exception:
                    continue

        source_context = "\n\n".join(source_texts) if source_texts else "(no source context available)"

        # Build prompt
        prompt_template = (Path(__file__).parent / "prompts" / "semantic_enrichment.txt").read_text()
        prompt = (
            prompt_template.replace("{claim_a}", claim_a)
            .replace("{claim_b}", claim_b)
            .replace("{content}", artifact.content)
            .replace("{source_context}", source_context)
        )

        try:
            response = ctx.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=temperature,
                artifact_desc=f"enrichment fix for {violation.artifact_id}",
            )
            response_text = response.content
        except Exception as exc:
            return FixAction(
                artifact_id=violation.artifact_id,
                action="skip",
                original_content_hash=artifact.content_hash,
                new_content="",
                new_content_hash="",
                description=f"LLM error: {exc}",
            )

        # Store trace
        _store_llm_trace(ctx.store, violation.artifact_id, prompt, response_text, "semantic_enrichment")

        # Parse response — handle both closed and unclosed code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)(?:\n?```|$)", response_text, re.DOTALL)
        text = match.group(1).strip() if match else response_text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return FixAction(
                artifact_id=violation.artifact_id,
                action="unresolved",
                original_content_hash=artifact.content_hash,
                new_content="",
                new_content_hash="",
                description="Could not parse LLM response",
                llm_explanation=response_text[:500],
            )

        status = data.get("status", "unresolved")
        new_content = data.get("content", "")
        explanation = data.get("explanation", "")

        if status == "resolved" and new_content:
            new_hash = f"sha256:{hashlib.sha256(new_content.encode()).hexdigest()}"
            return FixAction(
                artifact_id=violation.artifact_id,
                action="rewrite",
                original_content_hash=artifact.content_hash,
                new_content=new_content,
                new_content_hash=new_hash,
                description=explanation,
                evidence_source_ids=evidence_source_ids,
                interactive=True,
                llm_explanation=explanation,
            )
        else:
            return FixAction(
                artifact_id=violation.artifact_id,
                action="unresolved",
                original_content_hash=artifact.content_hash,
                new_content="",
                new_content_hash="",
                description=explanation or "Could not resolve contradiction",
                llm_explanation=explanation,
            )
