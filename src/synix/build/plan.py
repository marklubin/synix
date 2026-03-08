"""Dry-run build planning — analyze what would be built without executing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from synix.build.dag import compute_levels, resolve_build_order
from synix.build.fingerprint import Fingerprint
from synix.build.refs import synix_dir_for_build_dir
from synix.build.search_surfaces import transform_runtime_search_updates, validate_search_surface_uses
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.core.config import LLMConfig, redact_api_key
from synix.core.models import (
    Artifact,
    FlatFile,
    Layer,
    Pipeline,
    SearchIndex,
    SearchSurface,
    Source,
    SynixSearch,
    Transform,
    TransformContext,
)

# Default token estimates per LLM call
DEFAULT_INPUT_TOKENS_PER_CALL = 2000
DEFAULT_OUTPUT_TOKENS_PER_CALL = 500

# Default per-token pricing (USD) — Sonnet-class model
DEFAULT_INPUT_TOKEN_PRICE = 3.0 / 1_000_000  # $3 per million input tokens
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
    rebuild_count: int = 0  # artifacts that actually need rebuilding
    cached_count: int = 0  # artifacts that are already up-to-date
    resolved_llm_config: dict | None = None  # resolved LLM config for this layer (None for parse layers)
    parallel_units: int = 1  # number of parallel work units from split()
    fingerprint: dict | None = None  # current transform fingerprint (for --explain-cache)
    source_info: str | None = None  # e.g. "./sources/bios (3 .md files)" for parse layers


@dataclass
class ProjectionPlan:
    """Plan for a single projection."""

    name: str
    projection_type: str
    source_layers: list[str]
    status: str  # "cached", "rebuild", "new"
    artifact_count: int
    reason: str
    embedding_config: dict | None = None


@dataclass
class BuildPlan:
    """Complete build plan for a pipeline."""

    pipeline_name: str
    steps: list[StepPlan] = field(default_factory=list)
    surfaces: list[ProjectionPlan] = field(default_factory=list)
    projections: list[ProjectionPlan] = field(default_factory=list)
    total_estimated_llm_calls: int = 0
    total_estimated_tokens: int = 0
    total_estimated_cost: float = 0.0
    total_cached: int = 0
    total_rebuild: int = 0
    global_llm_config: dict = field(default_factory=dict)  # pipeline-level LLM config

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def plan_build(
    pipeline: Pipeline,
    source_dir: str | None = None,
    *,
    input_tokens_per_call: int = DEFAULT_INPUT_TOKENS_PER_CALL,
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_CALL,
    input_token_price: float = DEFAULT_INPUT_TOKEN_PRICE,
    output_token_price: float = DEFAULT_OUTPUT_TOKEN_PRICE,
) -> BuildPlan:
    """Walk the DAG and determine what would be built without executing."""
    src_dir = source_dir or pipeline.source_dir
    build_dir = Path(pipeline.build_dir)

    # Compute levels from DAG structure
    compute_levels(pipeline.layers)
    validate_search_surface_uses(pipeline)

    synix_dir = synix_dir_for_build_dir(build_dir)
    store = SnapshotArtifactCache(synix_dir) if synix_dir.exists() else None

    build_order = resolve_build_order(pipeline)

    # Resolve and store the global LLM config (with API key redacted)
    global_llm = LLMConfig.from_dict(pipeline.llm_config) if pipeline.llm_config else LLMConfig()
    plan = BuildPlan(
        pipeline_name=pipeline.name,
        global_llm_config=_llm_config_to_display_dict(global_llm),
    )

    # Track artifacts per layer for downstream dependency analysis
    layer_artifacts: dict[str, list[Artifact]] = {}

    for layer in build_order:
        step = _plan_layer(
            layer,
            pipeline,
            src_dir,
            store,
            layer_artifacts,
            plan.steps,
            input_tokens_per_call,
            output_tokens_per_call,
            input_token_price,
            output_token_price,
        )
        plan.steps.append(step)
        plan.total_estimated_llm_calls += step.estimated_llm_calls
        plan.total_estimated_tokens += step.estimated_tokens
        plan.total_estimated_cost += step.estimated_cost

        if step.status == "cached":
            plan.total_cached += 1
        else:
            plan.total_rebuild += 1

    # Plan build-time search surfaces
    for surface in pipeline.surfaces:
        surface_plan = _plan_projection(surface, pipeline, layer_artifacts, store)
        plan.surfaces.append(surface_plan)

    # Plan projections
    for proj in pipeline.projections:
        proj_plan = _plan_projection(proj, pipeline, layer_artifacts, store)
        plan.projections.append(proj_plan)

    return plan


def _plan_layer(
    layer: Layer,
    pipeline: Pipeline,
    src_dir: str,
    store: SnapshotArtifactCache | None,
    layer_artifacts: dict[str, list[Artifact]],
    prior_steps: list[StepPlan],
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    input_token_price: float,
    output_token_price: float,
) -> StepPlan:
    """Analyze a single layer to determine its build plan."""
    if isinstance(layer, Source):
        return _plan_source_layer(layer, pipeline, src_dir, store, layer_artifacts)

    # Must be a Transform
    assert isinstance(layer, Transform), f"Expected Transform, got {type(layer)}"

    # Gather inputs from dependent layers
    inputs: list[Artifact] = []
    for dep in layer.depends_on:
        dep_artifacts = layer_artifacts.get(dep.name)
        if dep_artifacts is None and store is not None:
            dep_artifacts = store.list_artifacts(dep.name)
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
    runtime_ctx = TransformContext.from_value(transform_config).with_updates(
        {
            "workspace": {
                "source_dir": src_dir,
                "build_dir": str(Path(pipeline.build_dir)),
            },
            **transform_runtime_search_updates(
                layer,
                build_dir=Path(pipeline.build_dir),
                projections=pipeline.projections,
            ),
        }
    )

    # Compute transform fingerprint
    transform_fp = layer.compute_fingerprint(transform_config)

    # Resolve the per-layer LLM config for display
    resolved_llm = _resolve_layer_llm_config(layer, pipeline)

    # Check if any upstream dependency has pending rebuilds
    upstream_dirty = False
    step_lookup = {s.name: s for s in prior_steps}
    for dep in layer.depends_on:
        dep_step = step_lookup.get(dep.name)
        if dep_step and dep_step.rebuild_count > 0:
            upstream_dirty = True
            break

    # For LLM layers, analyze cache state
    step = _plan_transform_layer(
        layer,
        inputs,
        store,
        layer_artifacts,
        upstream_dirty,
        input_tokens_per_call,
        output_tokens_per_call,
        input_token_price,
        output_token_price,
        transform_fp,
        transform_config,
    )
    step.resolved_llm_config = resolved_llm
    if transform_fp is not None:
        step.fingerprint = transform_fp.to_dict()

    # Estimate parallel unit count from split() if the layer will be built
    if step.status != "cached" and inputs:
        if upstream_dirty and hasattr(layer, "estimate_output_count"):
            step.parallel_units = layer.estimate_output_count(len(inputs))
        else:
            try:
                units = layer.split(inputs, runtime_ctx)
                step.parallel_units = len(units)
            except Exception:
                step.parallel_units = 1

    return step


def _compute_source_info(source_dir: str) -> str | None:
    """Compute human-readable source info string for a parse layer."""
    source_path = Path(source_dir)
    if not source_path.exists():
        return None

    from synix.adapters.registry import get_supported_extensions

    extensions = get_supported_extensions()

    # Count all files recursively (excluding hidden/dot files)
    all_files = [f for f in source_path.rglob("*") if f.is_file() and not f.name.startswith(".")]
    supported_files = [f for f in all_files if f.suffix.lower() in extensions]
    unsupported_count = len(all_files) - len(supported_files)

    # Count supported files by extension
    ext_counts: dict[str, int] = {}
    for f in supported_files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    if not ext_counts and unsupported_count == 0:
        return None

    parts = [f"{count} {ext}" for ext, count in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0]))]
    total = sum(ext_counts.values())
    if unsupported_count > 0:
        parts.append(f"{unsupported_count} unsupported")
        total += unsupported_count
    file_word = "file" if total == 1 else "files"

    # Note subdirectory depth when source spans nested dirs
    subdirs = [d for d in source_path.rglob("*") if d.is_dir() and not d.name.startswith(".")]
    depth_label = ""
    if subdirs:
        max_depth = max((len(d.relative_to(source_path).parts) for d in subdirs), default=0)
        depth_label = f", {max_depth} deep" if max_depth > 1 else ", nested"

    return f"{source_dir} ({', '.join(parts)} {file_word}{depth_label})"


def _plan_source_layer(
    layer: Source,
    pipeline: Pipeline,
    src_dir: str,
    store: SnapshotArtifactCache | None,
    layer_artifacts: dict[str, list[Artifact]],
) -> StepPlan:
    """Plan a Source layer by running its load() method."""
    source_dir = layer.dir or src_dir
    source_config = {"source_dir": source_dir}

    try:
        artifacts = layer.load(source_config)
    except Exception:
        # If loading fails (e.g., source_dir doesn't exist), report 0 artifacts
        artifacts = []

    layer_artifacts[layer.name] = artifacts
    artifact_count = len(artifacts)

    # Compute source info for display
    source_info = _compute_source_info(source_dir)

    if store is None:
        # No build dir yet -- everything is new
        return StepPlan(
            name=layer.name,
            level=layer._level,
            status="new",
            artifact_count=artifact_count,
            rebuild_count=artifact_count,
            cached_count=0,
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="new",
            source_info=source_info,
        )

    # Compare by label to detect modifications (same label, different artifact ID)
    existing = store.list_artifacts(layer.name)
    existing_by_label = {a.label: a.artifact_id for a in existing}
    new_by_label = {a.label: a.artifact_id for a in artifacts}

    added_labels = set(new_by_label) - set(existing_by_label)
    removed_labels = set(existing_by_label) - set(new_by_label)
    common_labels = set(new_by_label) & set(existing_by_label)
    modified_labels = {lbl for lbl in common_labels if new_by_label[lbl] != existing_by_label[lbl]}

    added = len(added_labels)
    removed = len(removed_labels)
    modified = len(modified_labels)
    cached_count = len(common_labels) - modified

    if added == 0 and removed == 0 and modified == 0:
        return StepPlan(
            name=layer.name,
            level=layer._level,
            status="cached",
            artifact_count=artifact_count,
            rebuild_count=0,
            cached_count=artifact_count,
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="all cached",
            source_info=source_info,
        )

    rebuild_count = modified + added

    if len(existing) == 0:
        status = "new"
        reason = "new"
    else:
        status = "rebuild"
        parts = []
        if modified > 0:
            parts.append(f"{modified} modified")
        if added > 0:
            parts.append(f"{added} added")
        if removed > 0:
            parts.append(f"{removed} removed")
        reason = ", ".join(parts) if parts else "sources changed"

    return StepPlan(
        name=layer.name,
        level=layer._level,
        status=status,
        artifact_count=artifact_count,
        rebuild_count=rebuild_count,
        cached_count=cached_count,
        estimated_llm_calls=0,
        estimated_tokens=0,
        estimated_cost=0.0,
        reason=reason,
        source_info=source_info,
    )


def _plan_transform_layer(
    layer: Transform,
    inputs: list[Artifact],
    store: SnapshotArtifactCache | None,
    layer_artifacts: dict[str, list[Artifact]],
    upstream_dirty: bool,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    input_token_price: float,
    output_token_price: float,
    transform_fp: Fingerprint | None = None,
    transform_config: dict | None = None,
) -> StepPlan:
    """Plan a Transform layer by checking cache state."""
    tokens_per_call = input_tokens_per_call + output_tokens_per_call
    cost_per_call = input_tokens_per_call * input_token_price + output_tokens_per_call * output_token_price

    if store is None:
        # No build dir -- everything is new
        estimated_count = layer.estimate_output_count(len(inputs))

        # Store placeholder artifacts for downstream planning
        layer_artifacts[layer.name] = inputs  # approximate

        return StepPlan(
            name=layer.name,
            level=layer._level,
            status="new",
            artifact_count=estimated_count,
            rebuild_count=estimated_count,
            cached_count=0,
            estimated_llm_calls=estimated_count,
            estimated_tokens=estimated_count * tokens_per_call,
            estimated_cost=estimated_count * cost_per_call,
            reason="new",
        )

    existing = store.list_artifacts(layer.name)

    # Check if the whole layer is fully cached
    fully_cached = not upstream_dirty and _is_layer_fully_cached(layer, existing, inputs, transform_fp)

    if fully_cached:
        layer_artifacts[layer.name] = existing
        return StepPlan(
            name=layer.name,
            level=layer._level,
            status="cached",
            artifact_count=len(existing),
            rebuild_count=0,
            cached_count=len(existing),
            estimated_llm_calls=0,
            estimated_tokens=0,
            estimated_cost=0.0,
            reason="all cached",
        )

    # Not fully cached — check per-artifact to get accurate rebuild count.
    global_change = _has_global_change(existing, transform_fp)

    rebuild_count = 0
    cached_count = 0

    if global_change:
        # Transform identity changed — all artifacts rebuild
        estimated_count = layer.estimate_output_count(len(inputs))
        rebuild_count = estimated_count
        cached_count = 0
    else:
        # Check per-artifact which are stale
        try:
            config = transform_config or {}
            runtime_ctx = TransformContext.from_value(config)
            units = layer.split(inputs, runtime_ctx)
            # Build a set of all input IDs covered by existing artifacts
            existing_by_inputs: dict[tuple[str, ...], bool] = {}
            for art in existing:
                key = tuple(sorted(art.input_ids))
                existing_by_inputs[key] = True

            for unit_inputs, _ in units:
                input_ids = tuple(sorted(a.artifact_id for a in unit_inputs))
                if input_ids in existing_by_inputs:
                    if upstream_dirty and len(unit_inputs) > 1:
                        # N:1 unit whose input IDs look cached, but upstream
                        # has pending rebuilds that will change those IDs.
                        rebuild_count += 1
                    else:
                        cached_count += 1
                else:
                    rebuild_count += 1
        except Exception:
            estimated_count = layer.estimate_output_count(len(inputs))
            rebuild_count = estimated_count
            cached_count = 0

    total_count = rebuild_count + cached_count

    # Determine reason for rebuild
    reason = _determine_rebuild_reason(existing, inputs, transform_fp)

    status = "rebuild" if existing else "new"
    if not existing:
        reason = "new"

    # Store existing artifacts for downstream planning
    layer_artifacts[layer.name] = existing if existing else inputs

    return StepPlan(
        name=layer.name,
        level=layer._level,
        status=status,
        artifact_count=total_count,
        rebuild_count=rebuild_count,
        cached_count=cached_count,
        estimated_llm_calls=rebuild_count,
        estimated_tokens=rebuild_count * tokens_per_call,
        estimated_cost=rebuild_count * cost_per_call,
        reason=reason,
    )


def _is_layer_fully_cached(
    layer: Layer,
    existing: list[Artifact],
    inputs: list[Artifact],
    transform_fp: Fingerprint | None = None,
) -> bool:
    """Check if a layer is fully cached — same logic as runner._layer_fully_cached."""
    if not existing:
        return False

    # Check transform identity via fingerprint
    if transform_fp is not None:
        for art in existing:
            stored_tfp_data = art.metadata.get("transform_fingerprint")
            if stored_tfp_data is None:
                return False
            stored_tfp = Fingerprint.from_dict(stored_tfp_data)
            if not transform_fp.matches(stored_tfp):
                return False

    # Check that all current inputs are covered by existing artifacts
    covered_input_ids: set[str] = set()
    for art in existing:
        covered_input_ids.update(art.input_ids)

    current_input_ids = {a.artifact_id for a in inputs}
    if not current_input_ids.issubset(covered_input_ids):
        return False

    return True


def _has_global_change(
    existing: list[Artifact],
    transform_fp: Fingerprint | None = None,
) -> bool:
    """Check if transform identity changed (affects all artifacts)."""
    if not existing:
        return False

    if transform_fp is not None:
        for art in existing:
            stored_tfp_data = art.metadata.get("transform_fingerprint")
            if stored_tfp_data is None:
                return True
            stored_tfp = Fingerprint.from_dict(stored_tfp_data)
            if not transform_fp.matches(stored_tfp):
                return True

    return False


def _determine_rebuild_reason(
    existing: list[Artifact],
    inputs: list[Artifact],
    transform_fp: Fingerprint | None = None,
) -> str:
    """Determine why a layer needs rebuild."""
    if not existing:
        return "new"

    reasons: list[str] = []

    # Check transform identity via fingerprint
    if transform_fp is not None:
        for art in existing:
            stored_tfp_data = art.metadata.get("transform_fingerprint")
            if stored_tfp_data is None:
                reasons.append("no stored fingerprint")
                break
            stored_tfp = Fingerprint.from_dict(stored_tfp_data)
            if stored_tfp is not None and not transform_fp.matches(stored_tfp):
                fp_reasons = transform_fp.explain_diff(stored_tfp)
                reasons.extend(fp_reasons)
                break

    # Check input changes
    covered_input_ids: set[str] = set()
    for art in existing:
        covered_input_ids.update(art.input_ids)
    current_input_ids = {a.artifact_id for a in inputs}
    if not current_input_ids.issubset(covered_input_ids):
        new_inputs = current_input_ids - covered_input_ids
        reasons.append(f"{len(new_inputs)} new input(s)")

    return ", ".join(reasons) if reasons else "inputs changed"


def _resolve_layer_llm_config(layer: Transform, pipeline: Pipeline) -> dict | None:
    """Resolve the effective LLM config for a layer by deep-merging over pipeline defaults.

    Returns None for Source layers which don't use LLM.
    For Transform layers, returns the resolved config dict with API key redacted.
    """
    # Start with pipeline defaults
    base = dict(pipeline.llm_config) if pipeline.llm_config else {}

    # Deep-merge layer-specific llm_config overrides
    if layer.config:
        layer_llm = layer.config.get("llm_config")
        if layer_llm:
            base.update(layer_llm)

    # Resolve into LLMConfig for display
    resolved = LLMConfig.from_dict(base)
    return _llm_config_to_display_dict(resolved)


def _llm_config_to_display_dict(config: LLMConfig) -> dict:
    """Convert an LLMConfig to a display-safe dict with redacted API key."""
    resolved_key = config.resolve_api_key()
    d: dict = {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.base_url:
        d["base_url"] = config.base_url
    if resolved_key:
        d["api_key"] = redact_api_key(resolved_key)
    return d


def _plan_projection(
    proj,
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
    store: SnapshotArtifactCache | None,
) -> ProjectionPlan:
    """Analyze a projection to determine its plan status.

    Projections are recorded as declarations in the manifest during build.
    Materialization happens at release time via ``synix release``.
    Search surfaces are materialized at build time to .synix/work/.
    """
    source_layers = [s.name for s in proj.sources]

    # Extract embedding config for display
    embedding_config = proj.embedding_config if isinstance(proj, (SearchIndex, SynixSearch, SearchSurface)) else None

    # Count total artifacts that would feed this projection
    all_artifacts: list[Artifact] = []
    for layer_name in source_layers:
        arts = layer_artifacts.get(layer_name, [])
        if not arts and store is not None:
            arts = store.list_artifacts(layer_name)
        all_artifacts.extend(arts)

    artifact_count = len(all_artifacts)

    # Determine projection type
    if isinstance(proj, SynixSearch):
        proj_type = "synix_search"
    elif isinstance(proj, SearchIndex):
        proj_type = "search_index"
    elif isinstance(proj, SearchSurface):
        proj_type = "search_surface"
    elif isinstance(proj, FlatFile):
        proj_type = "flat_file"
    else:
        proj_type = "unknown"

    # Surfaces are materialized at build time; projections are declared only
    if isinstance(proj, SearchSurface):
        return ProjectionPlan(
            name=proj.name,
            projection_type=proj_type,
            source_layers=source_layers,
            status="rebuild",
            artifact_count=artifact_count,
            reason="build-time surface",
            embedding_config=embedding_config,
        )

    return ProjectionPlan(
        name=proj.name,
        projection_type=proj_type,
        source_layers=source_layers,
        status="declared",
        artifact_count=artifact_count,
        reason="manifest declaration (materialize with synix release)",
        embedding_config=embedding_config,
    )
