"""Dry-run build planning — analyze what would be built without executing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from synix.core.models import Artifact, Layer, Pipeline
from synix.build.artifacts import ArtifactStore
from synix.build.dag import needs_rebuild, resolve_build_order
from synix.build.transforms import get_transform

# Import transform modules to trigger @register_transform decorators
import synix.build.parse_transform  # noqa: F401
import synix.build.llm_transforms  # noqa: F401

# Default token estimates per LLM call
DEFAULT_INPUT_TOKENS_PER_CALL = 2000
DEFAULT_OUTPUT_TOKENS_PER_CALL = 500

# Default per-token pricing (USD) — Sonnet-class model
DEFAULT_INPUT_TOKEN_PRICE = 3.0 / 1_000_000   # $3 per million input tokens
DEFAULT_OUTPUT_TOKEN_PRICE = 15.0 / 1_000_000  # $15 per million output tokens


@dataclass
class StepPlan:
    """Plan for a single pipeline layer/step."""

    name: str
    level: int
    status: str  # "rebuild", "cached", "new"
    artifact_count: int  # number of artifacts in this layer
    estimated_llm_calls: int  # 0 if cached
    estimated_tokens: int  # rough estimate (input + output)
    estimated_cost: float  # rough USD estimate
    reason: str  # why rebuild needed (e.g., "prompt changed", "new inputs", "all cached")


@dataclass
class BuildPlan:
    """Complete build plan for a pipeline."""

    pipeline_name: str
    steps: list[StepPlan] = field(default_factory=list)
    total_estimated_llm_calls: int = 0
    total_estimated_tokens: int = 0
    total_estimated_cost: float = 0.0
    total_cached: int = 0
    total_rebuild: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _transform_to_prompt_name(transform_name: str) -> str:
    """Map transform name to prompt template filename.

    Mirrors the mapping in runner.py.
    """
    mapping = {
        "episode_summary": "episode_summary",
        "monthly_rollup": "monthly_rollup",
        "topical_rollup": "topical_rollup",
        "core_synthesis": "core_memory",
    }
    return mapping.get(transform_name, transform_name)


def plan_build(
    pipeline: Pipeline,
    source_dir: str | None = None,
    *,
    input_tokens_per_call: int = DEFAULT_INPUT_TOKENS_PER_CALL,
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_CALL,
    input_token_price: float = DEFAULT_INPUT_TOKEN_PRICE,
    output_token_price: float = DEFAULT_OUTPUT_TOKEN_PRICE,
) -> BuildPlan:
    """Walk the DAG and determine what would be built without executing.

    This mirrors the runner logic but never calls the LLM. For level-0 (parse)
    layers, the parse transform is actually executed to count source artifacts
    (it is fast and does not use the LLM). For level > 0 layers, cache state
    is analyzed using the same needs_rebuild logic as the runner.

    Args:
        pipeline: The pipeline definition.
        source_dir: Override for pipeline.source_dir.
        input_tokens_per_call: Estimated input tokens per LLM call.
        output_tokens_per_call: Estimated output tokens per LLM call.
        input_token_price: Price per input token in USD.
        output_token_price: Price per output token in USD.

    Returns:
        BuildPlan with per-step analysis and cost estimates.
    """
    src_dir = source_dir or pipeline.source_dir
    build_dir = Path(pipeline.build_dir)

    store = ArtifactStore(build_dir) if build_dir.exists() else None

    build_order = resolve_build_order(pipeline)

    plan = BuildPlan(pipeline_name=pipeline.name)

    # Track artifacts per layer for downstream dependency analysis
    layer_artifacts: dict[str, list[Artifact]] = {}

    for layer in build_order:
        step = _plan_layer(
            layer, pipeline, src_dir, store, layer_artifacts,
            input_tokens_per_call, output_tokens_per_call,
            input_token_price, output_token_price,
        )
        plan.steps.append(step)
        plan.total_estimated_llm_calls += step.estimated_llm_calls
        plan.total_estimated_tokens += step.estimated_tokens
        plan.total_estimated_cost += step.estimated_cost

        if step.status == "cached":
            plan.total_cached += 1
        else:
            plan.total_rebuild += 1

    return plan


def _plan_layer(
    layer: Layer,
    pipeline: Pipeline,
    src_dir: str,
    store: ArtifactStore | None,
    layer_artifacts: dict[str, list[Artifact]],
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    input_token_price: float,
    output_token_price: float,
) -> StepPlan:
    """Analyze a single layer to determine its build plan."""
    # Gather inputs from dependent layers
    inputs: list[Artifact] = []
    if layer.depends_on:
        for dep_name in layer.depends_on:
            dep_artifacts = layer_artifacts.get(dep_name)
            if dep_artifacts is None and store is not None:
                dep_artifacts = store.list_artifacts(dep_name)
            if dep_artifacts:
                inputs.extend(dep_artifacts)

    # Build config for the transform (mirroring runner.py)
    base_llm_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
    transform_config: dict = {}
    transform_config["llm_config"] = dict(base_llm_config)
    transform_config["source_dir"] = src_dir
    if layer.context_budget is not None:
        transform_config["context_budget"] = layer.context_budget
    if layer.config:
        layer_llm = layer.config.get("llm_config")
        layer_rest = {k: v for k, v in layer.config.items() if k != "llm_config"}
        transform_config.update(layer_rest)
        if layer_llm:
            transform_config["llm_config"].update(layer_llm)

    transform_config["search_db_path"] = str(Path(pipeline.build_dir) / "search.db")

    # Get the transform
    transform = get_transform(layer.transform)
    prompt_id = None
    if layer.level > 0:
        try:
            prompt_id = transform.get_prompt_id(
                _transform_to_prompt_name(layer.transform)
            )
        except (FileNotFoundError, OSError):
            prompt_id = layer.transform

    transform_cache_key = transform.get_cache_key(transform_config)
    model_config = transform_config.get("llm_config", {}) if layer.level > 0 else None

    if layer.level == 0:
        # Parse layer: actually run the parse transform to count artifacts (fast, no LLM)
        return _plan_parse_layer(
            layer, transform, transform_config, store, layer_artifacts
        )

    # For LLM layers, analyze cache state
    return _plan_llm_layer(
        layer, inputs, prompt_id, model_config, transform_cache_key,
        store, layer_artifacts,
        input_tokens_per_call, output_tokens_per_call,
        input_token_price, output_token_price,
    )


def _plan_parse_layer(
    layer: Layer,
    transform,
    transform_config: dict,
    store: ArtifactStore | None,
    layer_artifacts: dict[str, list[Artifact]],
) -> StepPlan:
    """Plan a parse (level 0) layer by running the parse transform."""
    try:
        artifacts = transform.execute([], transform_config)
    except Exception:
        # If parsing fails (e.g., source_dir doesn't exist), report 0 artifacts
        artifacts = []

    layer_artifacts[layer.name] = artifacts
    artifact_count = len(artifacts)

    if store is None:
        # No build dir yet -- everything is new
        return StepPlan(
            name=layer.name,
            level=layer.level,
            status="new",
            artifact_count=artifact_count,
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="no previous build",
        )

    # Check how many are cached vs new
    existing = store.list_artifacts(layer.name)
    existing_hashes = {a.content_hash for a in existing}
    new_hashes = {a.content_hash for a in artifacts}

    new_count = len(new_hashes - existing_hashes)
    removed_count = len(existing_hashes - new_hashes)

    if new_count == 0 and removed_count == 0 and len(existing) == len(artifacts):
        return StepPlan(
            name=layer.name,
            level=layer.level,
            status="cached",
            artifact_count=artifact_count,
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="all cached",
        )

    if len(existing) == 0:
        status = "new"
        reason = "no previous build"
    else:
        status = "rebuild"
        parts = []
        if new_count > 0:
            parts.append(f"{new_count} new source(s)")
        if removed_count > 0:
            parts.append(f"{removed_count} removed source(s)")
        reason = ", ".join(parts) if parts else "sources changed"

    return StepPlan(
        name=layer.name,
        level=layer.level,
        status=status,
        artifact_count=artifact_count,
        estimated_llm_calls=0,
        estimated_tokens=0,
        estimated_cost=0.0,
        reason=reason,
    )


def _plan_llm_layer(
    layer: Layer,
    inputs: list[Artifact],
    prompt_id: str | None,
    model_config: dict | None,
    transform_cache_key: str,
    store: ArtifactStore | None,
    layer_artifacts: dict[str, list[Artifact]],
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    input_token_price: float,
    output_token_price: float,
) -> StepPlan:
    """Plan an LLM (level > 0) layer by checking cache state."""
    if store is None:
        # No build dir -- everything is new
        # Estimate artifact count from inputs and grouping
        estimated_count = _estimate_artifact_count(layer, inputs)
        tokens_per_call = input_tokens_per_call + output_tokens_per_call
        cost_per_call = (
            input_tokens_per_call * input_token_price
            + output_tokens_per_call * output_token_price
        )

        # Store placeholder artifacts for downstream planning
        layer_artifacts[layer.name] = inputs  # approximate

        return StepPlan(
            name=layer.name,
            level=layer.level,
            status="new",
            artifact_count=estimated_count,
            estimated_llm_calls=estimated_count,
            estimated_tokens=estimated_count * tokens_per_call,
            estimated_cost=estimated_count * cost_per_call,
            reason="no previous build",
        )

    existing = store.list_artifacts(layer.name)

    # Check if the whole layer is fully cached (same logic as runner._layer_fully_cached)
    fully_cached = _is_layer_fully_cached(
        layer, existing, inputs, prompt_id, model_config, transform_cache_key
    )

    if fully_cached:
        layer_artifacts[layer.name] = existing
        return StepPlan(
            name=layer.name,
            level=layer.level,
            status="cached",
            artifact_count=len(existing),
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="all cached",
        )

    # Determine reason for rebuild
    reason = _determine_rebuild_reason(
        existing, inputs, prompt_id, model_config, transform_cache_key
    )

    # Estimate count and cost
    estimated_count = _estimate_artifact_count(layer, inputs)
    # If existing artifacts exist, some might still be cached individually
    # but for planning, we report the full rebuild as worst-case
    rebuild_count = estimated_count

    tokens_per_call = input_tokens_per_call + output_tokens_per_call
    cost_per_call = (
        input_tokens_per_call * input_token_price
        + output_tokens_per_call * output_token_price
    )

    status = "rebuild" if existing else "new"
    if not existing:
        reason = "no previous build"

    # Store existing artifacts for downstream planning (they will be rebuilt, but
    # downstream layers need something for estimation)
    layer_artifacts[layer.name] = existing if existing else inputs

    return StepPlan(
        name=layer.name,
        level=layer.level,
        status=status,
        artifact_count=estimated_count,
        estimated_llm_calls=rebuild_count,
        estimated_tokens=rebuild_count * tokens_per_call,
        estimated_cost=rebuild_count * cost_per_call,
        reason=reason,
    )


def _is_layer_fully_cached(
    layer: Layer,
    existing: list[Artifact],
    inputs: list[Artifact],
    prompt_id: str | None,
    model_config: dict | None,
    transform_cache_key: str,
) -> bool:
    """Check if a layer is fully cached — same logic as runner._layer_fully_cached."""
    if not existing:
        return False

    for art in existing:
        if art.prompt_id != prompt_id:
            return False
        if model_config is not None and (art.model_config or {}) != model_config:
            return False
        if transform_cache_key:
            if art.metadata.get("transform_cache_key", "") != transform_cache_key:
                return False

    covered_input_hashes: set[str] = set()
    for art in existing:
        covered_input_hashes.update(art.input_hashes)

    current_input_hashes = {a.content_hash for a in inputs}
    if not current_input_hashes.issubset(covered_input_hashes):
        return False

    return True


def _determine_rebuild_reason(
    existing: list[Artifact],
    inputs: list[Artifact],
    prompt_id: str | None,
    model_config: dict | None,
    transform_cache_key: str,
) -> str:
    """Determine why a layer needs rebuild."""
    if not existing:
        return "no previous build"

    reasons = []

    # Check prompt change
    prompt_mismatch = any(art.prompt_id != prompt_id for art in existing)
    if prompt_mismatch:
        reasons.append("prompt changed")

    # Check model config change
    if model_config is not None:
        config_mismatch = any(
            (art.model_config or {}) != model_config for art in existing
        )
        if config_mismatch:
            reasons.append("model config changed")

    # Check transform cache key change
    if transform_cache_key:
        key_mismatch = any(
            art.metadata.get("transform_cache_key", "") != transform_cache_key
            for art in existing
        )
        if key_mismatch:
            reasons.append("transform config changed")

    # Check input changes
    covered_input_hashes: set[str] = set()
    for art in existing:
        covered_input_hashes.update(art.input_hashes)
    current_input_hashes = {a.content_hash for a in inputs}
    if not current_input_hashes.issubset(covered_input_hashes):
        new_inputs = current_input_hashes - covered_input_hashes
        reasons.append(f"{len(new_inputs)} new input(s)")

    return ", ".join(reasons) if reasons else "inputs changed"


def _estimate_artifact_count(layer: Layer, inputs: list[Artifact]) -> int:
    """Estimate how many artifacts a layer will produce based on grouping."""
    grouping = layer.grouping

    if grouping == "single":
        return 1
    elif grouping == "by_conversation":
        return len(inputs)
    elif grouping == "by_month":
        months = set()
        for art in inputs:
            month = art.metadata.get("date", "")[:7]
            if month and "-" in month:
                months.add(month)
            else:
                months.add("undated")
        return max(len(months), 1)
    elif grouping == "by_topic":
        topics = layer.config.get("topics", [])
        return max(len(topics), 1)
    else:
        # Default: one artifact per input
        return max(len(inputs), 1)
