"""Pluggable validator framework for domain-specific build validation.

Provides an ABC + decorator registry pattern (matching transforms.py) for
user-defined validators that return structured Violation objects with
automatic provenance tracing.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.core.citations import extract_citations
from synix.core.errors import atomic_write
from synix.core.models import Artifact, Pipeline

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceStep:
    """A single node in a provenance trace with field value at that node."""

    label: str
    layer: str
    field_value: str | None = None


@dataclass
class Violation:
    """A structured validation failure."""

    violation_type: str  # "mutual_exclusion", "required_field", custom
    severity: str  # "error", "warning"
    message: str  # human-readable summary
    label: str  # artifact that violated
    field: str  # metadata field involved
    metadata: dict = field(default_factory=dict)
    provenance_trace: list[ProvenanceStep] = field(default_factory=list)
    violation_id: str = ""  # Deterministic ID for dedup; computed at creation


@dataclass
class ValidationContext:
    """Context passed to validators — provides access to store, provenance, pipeline."""

    store: ArtifactStore
    provenance: ProvenanceTracker
    pipeline: Pipeline | None = None

    def trace_field_origin(self, label: str, field_name: str) -> list[ProvenanceStep]:
        """BFS walk provenance, collecting the field value at each node.

        Returns a list of ProvenanceStep from the starting artifact down
        to leaf ancestors, recording metadata[field_name] at each node.
        """
        steps: list[ProvenanceStep] = []
        visited: set[str] = set()
        queue: deque[str] = deque([label])

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            artifact = self.store.load_artifact(current_id)
            layer = ""
            field_value = None
            if artifact is not None:
                layer = artifact.metadata.get("layer_name", artifact.artifact_type)
                raw = artifact.metadata.get(field_name)
                if raw is not None:
                    field_value = str(raw) if not isinstance(raw, list) else str(raw)

            steps.append(
                ProvenanceStep(
                    label=current_id,
                    layer=layer,
                    field_value=field_value,
                )
            )

            # Walk to parents
            parent_ids = self.provenance.get_parents(current_id)
            for pid in parent_ids:
                if pid not in visited:
                    queue.append(pid)

        return steps


@dataclass
class ValidationResult:
    """Aggregated result of running all validators."""

    violations: list[Violation] = field(default_factory=list)
    validators_run: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "validators_run": self.validators_run,
            "violations": [
                {
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "message": v.message,
                    "label": v.label,
                    "field": v.field,
                    "metadata": v.metadata,
                    "provenance_trace": [
                        {
                            "label": s.label,
                            "layer": s.layer,
                            "field_value": s.field_value,
                        }
                        for s in v.provenance_trace
                    ],
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# Violation factory functions
# ---------------------------------------------------------------------------


def mutual_exclusion_violation(
    label: str,
    field_name: str,
    values: list[str],
    *,
    severity: str = "error",
    message: str | None = None,
) -> Violation:
    """Create a mutual exclusion violation."""
    msg = message or (f"Mutual exclusion on '{field_name}': {values}")
    return Violation(
        violation_type="mutual_exclusion",
        severity=severity,
        message=msg,
        label=label,
        field=field_name,
        metadata={"conflicting_values": values},
    )


def required_field_violation(
    label: str,
    field_name: str,
    *,
    severity: str = "error",
    message: str | None = None,
) -> Violation:
    """Create a required field violation."""
    msg = message or f"Missing required field '{field_name}'"
    return Violation(
        violation_type="required_field",
        severity=severity,
        message=msg,
        label=label,
        field=field_name,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_violation_id(violation_type: str, label: str, claim_a: str = "", claim_b: str = "") -> str:
    """Stable ID for deduplication. Normalized: lowercase, stripped, sorted claims."""
    claims = sorted([claim_a.strip().lower(), claim_b.strip().lower()])
    raw = f"{violation_type}|{label}|{claims[0]}|{claims[1]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _store_llm_trace(store: ArtifactStore, label: str, prompt: str, response: str, trace_type: str) -> None:
    """Save an LLM interaction as a system artifact for auditability."""
    trace = Artifact(
        label=f"trace-{trace_type}-{label}-{uuid4().hex[:8]}",
        artifact_type="llm_trace",
        content=json.dumps(
            {
                "trace_type": trace_type,
                "target_artifact": label,
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        ),
        metadata={"trace_type": trace_type, "target_artifact": label},
    )
    store.save_artifact(trace, layer_name="traces", layer_level=99)


# ---------------------------------------------------------------------------
# ViolationQueue
# ---------------------------------------------------------------------------


@dataclass
class ViolationQueue:
    """Persistent violation state + append-only audit log."""

    build_dir: Path
    _state: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.build_dir = Path(self.build_dir)

    @property
    def _state_path(self) -> Path:
        return self.build_dir / "violations_state.json"

    @property
    def _log_path(self) -> Path:
        return self.build_dir / "violations.jsonl"

    def save_state(self) -> None:
        atomic_write(self._state_path, json.dumps(self._state, indent=2, default=str))

    @classmethod
    def load(cls, build_dir) -> ViolationQueue:
        q = cls(build_dir=Path(build_dir))
        if q._state_path.exists():
            q._state = json.loads(q._state_path.read_text())
        return q

    def upsert(self, violation: Violation) -> None:
        vid = violation.violation_id
        if not vid:
            vid = compute_violation_id(violation.violation_type, violation.label)
        existing = self._state.get(vid)
        # Get artifact_id (hash) from violation metadata
        artifact_id = violation.metadata.get("artifact_id", "")
        self._state[vid] = {
            "violation_id": vid,
            "status": "active"
            if (
                existing is None or existing.get("status") != "ignored" or existing.get("last_seen_hash") != artifact_id
            )
            else "ignored",
            "last_seen_hash": artifact_id,
            "violation": {
                "violation_type": violation.violation_type,
                "severity": violation.severity,
                "message": violation.message,
                "label": violation.label,
                "field": violation.field,
                "metadata": violation.metadata,
                "violation_id": vid,
            },
        }
        # Reset ignored status if content changed
        if existing and existing.get("status") == "ignored" and existing.get("last_seen_hash") != artifact_id:
            self._state[vid]["status"] = "active"
        self._append_log(
            {
                "event": "detected",
                "violation_id": vid,
                "timestamp": datetime.now().isoformat(),
                "label": violation.label,
                "type": violation.violation_type,
            }
        )

    def ignore(self, violation_id: str) -> None:
        if violation_id in self._state:
            self._state[violation_id]["status"] = "ignored"
            self._append_log(
                {
                    "event": "ignored",
                    "violation_id": violation_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def resolve(self, violation_id: str, fix_action: str = "") -> None:
        if violation_id in self._state:
            self._state[violation_id]["status"] = "resolved"
            self._append_log(
                {
                    "event": "resolved",
                    "violation_id": violation_id,
                    "timestamp": datetime.now().isoformat(),
                    "fix_action": fix_action,
                }
            )

    def is_ignored(self, violation_id: str, content_hash: str) -> bool:
        entry = self._state.get(violation_id)
        if entry is None:
            return False
        if entry.get("status") != "ignored":
            return False
        # If content changed, ignore is invalidated
        return entry.get("last_seen_hash") == content_hash

    def active(self, store: ArtifactStore | None = None) -> list[dict]:
        result = []
        for entry in self._state.values():
            if entry.get("status") != "active":
                continue
            # Version check: if artifact was rebuilt, violation is stale
            if store is not None:
                label = entry["violation"].get("label", "")
                current_hash = store.get_artifact_id(label)
                if current_hash and current_hash != entry.get("last_seen_hash"):
                    entry["status"] = "expired"
                    self._append_log(
                        {
                            "event": "expired",
                            "violation_id": entry["violation_id"],
                            "timestamp": datetime.now().isoformat(),
                            "reason": "artifact_rebuilt",
                        }
                    )
                    continue
            result.append(entry["violation"])
        return result

    def _append_log(self, event: dict) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# BaseValidator ABC + registry
# ---------------------------------------------------------------------------


class BaseValidator(ABC):
    """Abstract base class for domain validators."""

    @abstractmethod
    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        """Validate a set of artifacts and return any violations found."""
        ...


_VALIDATORS: dict[str, type[BaseValidator]] = {}


def register_validator(name: str):
    """Decorator to register a validator class by name."""

    def wrapper(cls: type[BaseValidator]) -> type[BaseValidator]:
        _VALIDATORS[name] = cls
        return cls

    return wrapper


def get_validator(name: str) -> BaseValidator:
    """Get an instantiated validator by name."""
    if name not in _VALIDATORS:
        raise ValueError(f"Unknown validator: {name}. Available: {list(_VALIDATORS.keys())}")
    return _VALIDATORS[name]()


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------


@register_validator("mutual_exclusion")
class MutualExclusionValidator(BaseValidator):
    """Checks that merge artifacts don't mix values of a given metadata field.

    Config:
        field: metadata field to check (e.g. "customer_id")
        scope: artifact type prefix to filter (e.g. "merge")
    """

    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        violations: list[Violation] = []

        for artifact in artifacts:
            parent_ids = ctx.provenance.get_parents(artifact.label)
            values: set[str] = set()

            for pid in parent_ids:
                parent = ctx.store.load_artifact(pid)
                if parent is not None:
                    raw = parent.metadata.get(self._field_name, None)
                    if raw is not None:
                        values.add(str(raw))

            # Also check source_{field}s in the artifact's own metadata
            source_key = f"source_{self._field_name}s"
            source_vals = artifact.metadata.get(source_key, [])
            if isinstance(source_vals, list):
                for v in source_vals:
                    if v is not None:
                        values.add(str(v))

            if len(values) > 1:
                violations.append(
                    mutual_exclusion_violation(
                        label=artifact.label,
                        field_name=self._field_name,
                        values=sorted(values),
                    )
                )

        return violations


@register_validator("required_field")
class RequiredFieldValidator(BaseValidator):
    """Checks that artifacts in specified layers have a required metadata field.

    Config:
        field: metadata field to check (e.g. "customer_id")
        layers: list of layer names to check
    """

    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        violations: list[Violation] = []

        for artifact in artifacts:
            raw = artifact.metadata.get(self._field_name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                violations.append(
                    required_field_violation(
                        label=artifact.label,
                        field_name=self._field_name,
                    )
                )

        return violations


@register_validator("pii")
class PIIValidator(BaseValidator):
    """Detects PII patterns (credit cards, SSNs, emails, phone numbers)."""

    PATTERNS = {
        "credit_card": re.compile(
            r"\b(?:4[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}"  # Visa
            r"|5[1-5][0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}"  # MC
            r"|3[47][0-9]{1}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{3,4}"  # Amex
            r"|6(?:011|5[0-9]{2})[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})\b"  # Discover
        ),
        "ssn": re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
    }

    @staticmethod
    def _redact(value: str, pattern_name: str) -> str:
        """Redact a matched PII value for display."""
        if pattern_name == "credit_card":
            digits = re.sub(r"[\s-]", "", value)
            return digits[:4] + "****" + digits[-2:]
        elif pattern_name == "ssn":
            return "***-**-" + value[-4:]
        elif pattern_name == "email":
            parts = value.split("@")
            return parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else "***"
        elif pattern_name == "phone":
            return "***-***-" + value[-4:]
        return "****"

    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        violations: list[Violation] = []
        config = getattr(self, "_config", {})
        enabled_patterns = config.get("patterns", list(self.PATTERNS.keys()))
        severity = config.get("severity", "warning")

        for artifact in artifacts:
            for pattern_name in enabled_patterns:
                pattern = self.PATTERNS.get(pattern_name)
                if pattern is None:
                    continue
                matches = pattern.findall(artifact.content)
                for match in matches:
                    redacted = self._redact(match, pattern_name)
                    vid = compute_violation_id("pii", artifact.label, pattern_name, match)
                    violations.append(
                        Violation(
                            violation_type="pii",
                            severity=severity,
                            message=f"PII detected ({pattern_name}): {redacted}",
                            label=artifact.label,
                            field="content",
                            metadata={
                                "pattern": pattern_name,
                                "redacted_value": redacted,
                                "artifact_id": artifact.artifact_id,
                            },
                            violation_id=vid,
                        )
                    )
        return violations


# ---------------------------------------------------------------------------
# Semantic conflict helpers
# ---------------------------------------------------------------------------


def _parse_conflict_response(response_text: str) -> list[dict]:
    """Extract conflict list from LLM response (JSON in code blocks or raw)."""
    # Try to extract JSON from code blocks first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    text = match.group(1).strip() if match else response_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    conflicts = data.get("conflicts", [])
    if not isinstance(conflicts, list):
        return []

    # Normalize: ensure source hint fields have defaults
    for c in conflicts:
        c.setdefault("claim_a_source_hint", "")
        c.setdefault("claim_b_source_hint", "")
        c.setdefault("explanation", "")
        c.setdefault("confidence", "medium")

    return conflicts


@register_validator("semantic_conflict")
class SemanticConflictValidator(BaseValidator):
    """LLM-based validator that detects contradictions in synthesized artifacts.

    Config:
        layers: list of layer names to check
        llm_config: dict with LLM settings
        max_artifacts: max artifacts to check (default 20)
    """

    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        config = getattr(self, "_config", {})
        max_artifacts = config.get("max_artifacts", 20)

        violations: list[Violation] = []

        # Get LLM client: pre-built _llm_client (for testing) or create from llm_config
        client = config.get("_llm_client")
        if client is None:
            llm_config_dict = config.get("llm_config", {})
            try:
                from synix.build.cassette import maybe_wrap_client
                from synix.build.llm_client import LLMClient
                from synix.core.config import LLMConfig

                llm_cfg = LLMConfig.from_dict(llm_config_dict)
                client = maybe_wrap_client(LLMClient(llm_cfg))
            except Exception:
                return violations  # Can't validate without LLM

        # Load prompt template
        try:
            prompt_path = Path(__file__).parent / "prompts" / "semantic_conflict.txt"
            prompt_template = prompt_path.read_text()
        except (FileNotFoundError, OSError):
            return violations

        # Optional search index for claim tracing (lazy import to avoid build→search dep)
        search_index = None
        try:
            import importlib

            _indexer = importlib.import_module("synix.search.indexer")
            search_db = ctx.store.build_dir / "search.db"
            if search_db.exists():
                search_index = _indexer.SearchIndex(search_db)
        except Exception:
            pass

        for artifact in artifacts[:max_artifacts]:
            try:
                prompt = prompt_template.replace("{content}", artifact.content)
                response = client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    artifact_desc=f"conflict check for {artifact.label}",
                )

                # Store LLM trace
                _store_llm_trace(ctx.store, artifact.label, prompt, response.content, "semantic_conflict_check")

                conflicts = _parse_conflict_response(response.content)

                for conflict in conflicts:
                    title = conflict.get("title", "")
                    claim_a = conflict.get("claim_a", "")
                    claim_b = conflict.get("claim_b", "")
                    explanation = conflict.get("explanation", "")
                    confidence = conflict.get("confidence", "medium")
                    claim_a_hint = conflict.get("claim_a_source_hint", "")
                    claim_b_hint = conflict.get("claim_b_source_hint", "")

                    # Trace claims to source artifacts via search index
                    source_ids: list[str] = []
                    if search_index is not None:
                        for query_text in [claim_a, claim_b]:
                            if query_text:
                                try:
                                    results = search_index.query(query_text)
                                    for r in results[:3]:
                                        if r.label not in source_ids:
                                            source_ids.append(r.label)
                                except Exception:
                                    pass

                    vid = compute_violation_id("semantic_conflict", artifact.label, claim_a, claim_b)

                    severity = "error" if confidence == "high" else "warning"

                    message = title if title else (f'"{claim_a}" vs "{claim_b}"')

                    violations.append(
                        Violation(
                            violation_type="semantic_conflict",
                            severity=severity,
                            message=message,
                            label=artifact.label,
                            field="content",
                            metadata={
                                "title": title,
                                "claim_a": claim_a,
                                "claim_b": claim_b,
                                "claim_a_source_hint": claim_a_hint,
                                "claim_b_source_hint": claim_b_hint,
                                "explanation": explanation,
                                "confidence": confidence,
                                "source_ids": source_ids,
                                "artifact_id": artifact.artifact_id,
                            },
                            violation_id=vid,
                        )
                    )

            except Exception:
                # LLM errors → skip this artifact gracefully
                continue

        if search_index is not None:
            search_index.close()

        return violations


# ---------------------------------------------------------------------------
# Citation check helpers
# ---------------------------------------------------------------------------


def _parse_citation_response(response_text: str) -> list[dict]:
    """Extract ungrounded claims list from LLM response (JSON in code blocks or raw)."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    text = match.group(1).strip() if match else response_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    ungrounded = data.get("ungrounded", [])
    if not isinstance(ungrounded, list):
        return []

    for item in ungrounded:
        item.setdefault("claim", "")
        item.setdefault("suggestion", "")

    return ungrounded


@register_validator("citation")
class CitationValidator(BaseValidator):
    """LLM-based validator that checks whether claims are grounded by synix:// citations.

    Config:
        layers: list of layer names to check
        llm_config: dict with LLM settings
        max_artifacts: max artifacts to check (default 20)
    """

    def validate(self, artifacts: list[Artifact], ctx: ValidationContext) -> list[Violation]:
        config = getattr(self, "_config", {})
        max_artifacts = config.get("max_artifacts", 20)

        violations: list[Violation] = []

        # Get LLM client
        client = config.get("_llm_client")
        if client is None:
            llm_config_dict = config.get("llm_config", {})
            try:
                from synix.build.cassette import maybe_wrap_client
                from synix.build.llm_client import LLMClient
                from synix.core.config import LLMConfig

                llm_cfg = LLMConfig.from_dict(llm_config_dict)
                client = maybe_wrap_client(LLMClient(llm_cfg))
            except Exception:
                return violations

        # Load prompt template
        try:
            prompt_path = Path(__file__).parent / "prompts" / "citation_check.txt"
            prompt_template = prompt_path.read_text()
        except (FileNotFoundError, OSError):
            return violations

        for artifact in artifacts[:max_artifacts]:
            try:
                existing = extract_citations(artifact.content)
                existing_uris = ", ".join(c.uri for c in existing) if existing else "(none)"

                prompt = prompt_template.replace("{content}", artifact.content).replace(
                    "{existing_citations}", existing_uris
                )

                response = client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    artifact_desc=f"citation check for {artifact.label}",
                )

                _store_llm_trace(ctx.store, artifact.label, prompt, response.content, "citation_check")

                ungrounded = _parse_citation_response(response.content)

                for item in ungrounded:
                    claim = item.get("claim", "")
                    suggestion = item.get("suggestion", "")

                    vid = compute_violation_id("ungrounded_claim", artifact.label, claim)

                    violations.append(
                        Violation(
                            violation_type="ungrounded_claim",
                            severity="warning",
                            message=f"Ungrounded claim: {claim[:80]}",
                            label=artifact.label,
                            field="content",
                            metadata={
                                "claim": claim,
                                "suggestion": suggestion,
                                "existing_citations": [c.uri for c in existing],
                                "artifact_id": artifact.artifact_id,
                            },
                            violation_id=vid,
                        )
                    )

            except Exception:
                continue

        return violations


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def _gather_artifacts(store: ArtifactStore, config: dict) -> list[Artifact]:
    """Gather artifacts matching the validator config's scope/layers filters."""
    scope = config.get("scope")
    layers = config.get("layers")
    artifact_ids = config.get("artifact_ids")

    if artifact_ids:
        # Exact artifact ID list
        artifacts: list[Artifact] = []
        for aid in artifact_ids:
            art = store.load_artifact(aid)
            if art is not None:
                artifacts.append(art)
        return artifacts

    if layers:
        # Gather artifacts from specified layers
        artifacts = []
        for layer_name in layers:
            artifacts.extend(store.list_artifacts(layer_name))
        return artifacts

    if scope:
        # Gather artifacts whose label starts with scope prefix or artifact_type matches
        all_artifacts: list[Artifact] = []
        for aid in store._manifest:
            art = store.load_artifact(aid)
            if art is not None:
                if art.label.startswith(scope + "-") or art.artifact_type == scope:
                    all_artifacts.append(art)
        return all_artifacts

    # No filter — return all artifacts
    all_artifacts = []
    for aid in store._manifest:
        art = store.load_artifact(aid)
        if art is not None:
            all_artifacts.append(art)
    return all_artifacts


def run_validators(
    pipeline: Pipeline,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
) -> ValidationResult:
    """Run all validators declared in the pipeline.

    For each ValidatorDecl in pipeline.validators:
    1. Instantiate the registered validator
    2. Gather matching artifacts based on config (scope/layers)
    3. Run validation
    4. Auto-resolve provenance traces for violations that lack them

    Returns aggregated ValidationResult.
    """
    ctx = ValidationContext(store, provenance, pipeline)
    result = ValidationResult()

    for decl in pipeline.validators:
        validator = get_validator(decl.name)

        # Pass config field to the validator instance
        field_name = decl.config.get("field", "")
        validator._field_name = field_name  # type: ignore[attr-defined]
        validator._config = decl.config  # type: ignore[attr-defined]

        artifacts = _gather_artifacts(store, decl.config)
        violations = validator.validate(artifacts, ctx)

        # Auto-resolve provenance for violations without traces
        for v in violations:
            if not v.provenance_trace:
                v.provenance_trace = ctx.trace_field_origin(v.label, v.field)

        result.violations.extend(violations)
        result.validators_run.append(decl.name)

    return result
