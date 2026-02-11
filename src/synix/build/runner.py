"""Pipeline runner — walk DAG, run transforms, cache artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import synix.build.llm_transforms  # noqa: F401
import synix.build.merge_transform  # noqa: F401

# Import transform modules to trigger @register_transform decorators
import synix.build.parse_transform  # noqa: F401
from synix.build.artifacts import ArtifactStore
from synix.build.dag import needs_rebuild, resolve_build_order
from synix.build.projections import FlatFileProjection, get_projection
from synix.build.provenance import ProvenanceTracker
from synix.build.transforms import get_transform
from synix.core.logging import SynixLogger, Verbosity
from synix.core.models import Artifact, Layer, Pipeline, Projection


@dataclass
class LayerStats:
    """Build statistics for a single layer."""

    name: str
    level: int
    built: int = 0
    cached: int = 0
    skipped: int = 0
    time_seconds: float = 0.0


@dataclass
class ProjectionStats:
    """Build statistics for a single projection."""

    name: str
    status: str = "built"  # "built", "cached"


@dataclass
class RunResult:
    """Summary of a pipeline run."""

    built: int = 0
    cached: int = 0
    skipped: int = 0
    total_time: float = 0.0
    layer_stats: list[LayerStats] = field(default_factory=list)
    projection_stats: list[ProjectionStats] = field(default_factory=list)
    run_log: dict = field(default_factory=dict)
    validation: object | None = None  # ValidationResult when validators are declared


def run(pipeline: Pipeline, source_dir: str | None = None, verbosity: int = 0,
        concurrency: int = 5, progress=None, validate: bool = False) -> RunResult:
    """Execute the full pipeline — walk DAG, run transforms, materialize projections.

    Args:
        pipeline: The pipeline definition.
        source_dir: Override for pipeline.source_dir.
        verbosity: Verbosity level (0=default, 1=verbose, 2=debug).
        concurrency: Number of concurrent LLM requests (1 = sequential).
        progress: Optional BuildProgress for live display updates.
        validate: Run domain validators after build (default False).

    Returns:
        RunResult with build statistics.
    """
    start_time = time.time()
    src_dir = source_dir or pipeline.source_dir
    build_dir = Path(pipeline.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(build_dir)
    provenance = ProvenanceTracker(build_dir)
    result = RunResult()

    # Create structured logger
    logger = SynixLogger(
        verbosity=Verbosity(min(verbosity, Verbosity.DEBUG)),
        build_dir=build_dir,
        progress=progress,
    )

    # Resolve build order
    build_order = resolve_build_order(pipeline)
    logger.run_start(pipeline.name, len(build_order))

    # Artifacts produced per layer (layer_name -> list[Artifact])
    layer_artifacts: dict[str, list[Artifact]] = {}

    for layer in build_order:
        layer_start = time.time()
        stats = LayerStats(name=layer.name, level=layer.level)
        logger.layer_start(layer.name, layer.level)

        # Gather inputs from dependent layers
        inputs: list[Artifact] = []
        if layer.depends_on:
            for dep_name in layer.depends_on:
                dep_artifacts = layer_artifacts.get(dep_name)
                if dep_artifacts is None:
                    # Load from store if not in memory (e.g., cached from previous run)
                    dep_artifacts = store.list_artifacts(dep_name)
                inputs.extend(dep_artifacts)

        # Build config for the transform
        base_llm_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
        transform_config: dict = {}
        transform_config["llm_config"] = dict(base_llm_config)
        transform_config["source_dir"] = src_dir
        if layer.context_budget is not None:
            transform_config["context_budget"] = layer.context_budget
        if layer.config:
            # Deep-merge llm_config from layer config over pipeline defaults
            layer_llm = layer.config.get("llm_config")
            layer_rest = {k: v for k, v in layer.config.items() if k != "llm_config"}
            transform_config.update(layer_rest)
            if layer_llm:
                transform_config["llm_config"].update(layer_llm)

        # For topical rollup, pass the search index path
        transform_config["search_db_path"] = str(build_dir / "search.db")
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

        # Compute transform-specific cache key
        transform_cache_key = transform.get_cache_key(transform_config)
        # Only use model_config for cache comparison on LLM layers (level > 0).
        # Parse transforms (level 0) don't use LLM and store model_config=None.
        model_config = transform_config.get("llm_config", {}) if layer.level > 0 else None

        # For non-parse layers, check if we can skip the transform entirely.
        # If existing artifacts have matching input hashes, prompt_id, model_config,
        # and transform cache key, reuse them without calling the LLM.
        layer_built: list[Artifact] = []

        if layer.level > 0 and _layer_fully_cached(
            layer, inputs, prompt_id, model_config, transform_cache_key, store
        ):
            # All cached — load existing artifacts
            existing = store.list_artifacts(layer.name)
            for art in existing:
                art.metadata["layer_name"] = layer.name
                art.metadata["layer_level"] = layer.level
                layer_built.append(art)
                stats.cached += 1
                logger.artifact_cached(layer.name, art.artifact_id)
        else:
            # Execute transform to get candidate artifacts
            # Pass the logger to transforms via config so LLM calls can be tracked
            transform_config["_logger"] = logger
            transform_config["_layer_name"] = layer.name

            # Helper to save a single artifact immediately (save-as-you-go)
            def _save_artifact(artifact: Artifact) -> None:
                if needs_rebuild(
                    artifact.artifact_id, artifact.input_hashes, prompt_id, store,
                    current_model_config=model_config,
                    current_transform_cache_key=transform_cache_key,
                ):
                    artifact.metadata["layer_name"] = layer.name
                    artifact.metadata["layer_level"] = layer.level
                    if transform_cache_key:
                        artifact.metadata["transform_cache_key"] = transform_cache_key
                    store.save_artifact(artifact, layer.name, layer.level)
                    parent_ids = _get_parent_artifact_ids(artifact, inputs)
                    provenance.record(
                        artifact.artifact_id,
                        parent_ids=parent_ids,
                        prompt_id=artifact.prompt_id,
                        model_config=artifact.model_config,
                    )
                    layer_built.append(artifact)
                    stats.built += 1
                    logger.artifact_built(layer.name, artifact.artifact_id)
                else:
                    cached = store.load_artifact(artifact.artifact_id)
                    if cached is not None:
                        cached.metadata["layer_name"] = layer.name
                        cached.metadata["layer_level"] = layer.level
                        layer_built.append(cached)
                    else:
                        layer_built.append(artifact)
                    stats.cached += 1
                    logger.artifact_cached(layer.name, artifact.artifact_id)

            def _on_batch_complete(artifacts: list[Artifact]) -> None:
                """Callback for concurrent executor — save each artifact immediately."""
                for artifact in artifacts:
                    _save_artifact(artifact)

            # Split inputs into work units via the transform's split() method.
            # Parallelism is determined entirely by split() — transforms that
            # can't be parallelized return a single unit.
            units = transform.split(inputs, transform_config)
            use_concurrent = concurrency > 1 and len(units) > 1

            if use_concurrent:
                _execute_transform_concurrent(
                    transform, units, transform_config, concurrency,
                    on_complete=_on_batch_complete,
                )
            else:
                for unit_inputs, config_extras in units:
                    merged_config = {**transform_config, **config_extras}
                    new_artifacts = transform.execute(unit_inputs, merged_config)
                    for artifact in new_artifacts:
                        _save_artifact(artifact)

        layer_artifacts[layer.name] = layer_built

        stats.time_seconds = time.time() - layer_start
        result.layer_stats.append(stats)
        result.built += stats.built
        result.cached += stats.cached
        result.skipped += stats.skipped
        logger.layer_finish(layer.name, stats.built, stats.cached)

        # Materialize intermediate projections for this layer
        # (e.g., episode search index so topical rollup can query it)
        _materialize_layer_projections(
            pipeline, layer.name, layer_artifacts, store, build_dir, logger=logger
        )

    # Materialize all final projections (with caching)
    result.projection_stats = _materialize_all_projections(
        pipeline, layer_artifacts, store, build_dir, logger=logger
    )

    # Run domain validators if requested and declared
    if validate and pipeline.validators:
        from synix.build.validators import run_validators
        result.validation = run_validators(pipeline, store, provenance)

    result.total_time = time.time() - start_time
    logger.run_finish(result.total_time)
    result.run_log = logger.run_log.to_dict()
    return result


def _layer_fully_cached(
    layer: Layer,
    inputs: list[Artifact],
    prompt_id: str | None,
    model_config: dict | None,
    transform_cache_key: str,
    store: ArtifactStore,
) -> bool:
    """Check if a layer can be entirely skipped (all artifacts cached)."""
    existing = store.list_artifacts(layer.name)
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


def _get_parent_artifact_ids(artifact: Artifact, inputs: list[Artifact]) -> list[str]:
    """Determine parent artifact IDs based on input hashes."""
    hash_to_id: dict[str, str] = {}
    for inp in inputs:
        if inp.content_hash:
            hash_to_id[inp.content_hash] = inp.artifact_id

    parents = []
    for h in artifact.input_hashes:
        if h in hash_to_id:
            parents.append(hash_to_id[h])

    if not parents and inputs:
        parents = [inp.artifact_id for inp in inputs]

    return parents


def _execute_transform_concurrent(
    transform,
    units: list[tuple[list[Artifact], dict]],
    config: dict,
    concurrency: int,
    on_complete=None,
) -> list[Artifact]:
    """Execute transform work units concurrently.

    Each unit is (unit_inputs, config_extras) as returned by transform.split().
    Runs transform.execute(unit_inputs, merged_config) in parallel using a
    thread pool. Calls on_complete(artifacts) as each finishes so artifacts
    can be saved immediately (save-as-you-go).

    Args:
        transform: The transform instance to execute.
        units: List of (unit_inputs, config_extras) tuples from split().
        config: Base transform configuration dict.
        concurrency: Maximum number of concurrent workers.
        on_complete: Optional callback(list[Artifact]) called as each unit completes.

    Returns:
        List of output artifacts in the same order as units.
    """
    results: list[list[Artifact] | Exception] = [None] * len(units)  # type: ignore[list-item]

    # Keys that should be shared (not deep-copied) across workers because they
    # contain non-picklable objects (e.g. logger with file handles).
    _shared_keys = {"_logger"}

    # Pre-create a shared LLM client to avoid per-thread connection overhead
    # _make_llm_client already applies cassette wrapping
    shared_client = None
    try:
        from synix.build.llm_transforms import _make_llm_client
        shared_client = _make_llm_client(config)
    except Exception:
        pass  # Non-LLM transforms don't need a shared client

    def _run_one(index: int, unit_inputs: list[Artifact], config_extras: dict) -> tuple[int, list[Artifact]]:
        """Execute the transform for a single unit, returning (index, artifacts)."""
        # Deep-copy mutable config to avoid cross-thread mutation, but share
        # non-copyable objects like the logger.
        shared = {k: config[k] for k in _shared_keys if k in config}
        copyable = {k: v for k, v in config.items() if k not in _shared_keys}
        worker_config = copy.deepcopy(copyable)
        worker_config.update(shared)
        worker_config.update(config_extras)
        if shared_client is not None:
            worker_config["_shared_llm_client"] = shared_client
        return index, transform.execute(unit_inputs, worker_config)

    first_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, i, unit_inputs, config_extras): i
            for i, (unit_inputs, config_extras) in enumerate(units)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, artifacts = future.result()
                results[idx] = artifacts
                if on_complete:
                    on_complete(artifacts)
            except Exception as exc:
                results[idx] = exc
                if first_error is None:
                    first_error = exc

    # Flatten successful results in order
    all_artifacts: list[Artifact] = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        all_artifacts.extend(r)

    if first_error is not None:
        raise first_error

    return all_artifacts


def _transform_to_prompt_name(transform_name: str) -> str:
    """Map transform name to prompt template filename."""
    mapping = {
        "episode_summary": "episode_summary",
        "monthly_rollup": "monthly_rollup",
        "topical_rollup": "topical_rollup",
        "core_synthesis": "core_memory",
    }
    return mapping.get(transform_name, transform_name)


def _materialize_layer_projections(
    pipeline: Pipeline,
    layer_name: str,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
    build_dir: Path,
    logger: SynixLogger | None = None,
) -> None:
    """Materialize projections that source from this layer.

    For search_index projections: progressively materialize with whatever
    source layers are available so far (so downstream transforms can query).
    For flat_file projections: wait until all source layers are available.
    """
    for proj in pipeline.projections:
        source_layers = [s["layer"] for s in proj.sources]
        if layer_name not in source_layers:
            continue

        if proj.projection_type == "flat_file":
            # Flat file only makes sense with all sources (e.g., core layer)
            if not all(ln in layer_artifacts for ln in source_layers):
                continue
            _materialize_projection(proj, layer_artifacts, build_dir,
                                    logger=logger, triggered_by=layer_name)
        elif proj.projection_type == "search_index":
            # Progressive: materialize with whatever sources are available
            available_sources = [s for s in proj.sources if s["layer"] in layer_artifacts]
            if available_sources:
                _materialize_projection(
                    proj, layer_artifacts, build_dir,
                    source_override=available_sources, logger=logger,
                    triggered_by=layer_name,
                )
        else:
            # Unknown projection type — require all sources
            if not all(ln in layer_artifacts for ln in source_layers):
                continue
            _materialize_projection(proj, layer_artifacts, build_dir,
                                    logger=logger, triggered_by=layer_name)


def _materialize_all_projections(
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
    build_dir: Path,
    logger: SynixLogger | None = None,
) -> list[ProjectionStats]:
    """Materialize all projections after the full build (with caching)."""
    cache = _load_projection_cache(build_dir)
    stats: list[ProjectionStats] = []

    # Determine the last source layer for each projection (for triggered_by)
    build_order = resolve_build_order(pipeline)
    layer_order = {layer.name: i for i, layer in enumerate(build_order)}

    for proj in pipeline.projections:
        # Gather all artifacts for this projection
        all_artifacts: list[Artifact] = []
        for source in proj.sources:
            all_artifacts.extend(layer_artifacts.get(source["layer"], []))

        # Determine the last source layer (by build order) for triggered_by
        source_layers = [s["layer"] for s in proj.sources]
        last_layer = max(source_layers, key=lambda ln: layer_order.get(ln, 0)) if source_layers else None

        # Check cache — skip if hash matches (includes projection config)
        current_hash = _compute_projection_hash(all_artifacts, proj.config)
        cached_entry = cache.get(proj.name)
        if (
            cached_entry is not None
            and cached_entry.get("source_hash") == current_hash
            and cached_entry.get("artifact_count") == len(all_artifacts)
        ):
            if logger:
                logger.projection_cached(proj.name, triggered_by=last_layer)
            stats.append(ProjectionStats(name=proj.name, status="cached"))
            continue

        # Materialize
        cached = _materialize_projection(
            proj, layer_artifacts, build_dir, logger=logger,
            triggered_by=last_layer,
        )
        status = "cached" if cached else "built"
        stats.append(ProjectionStats(name=proj.name, status=status))

        # Update cache
        config_hash = (
            hashlib.sha256(json.dumps(proj.config, sort_keys=True, default=str).encode()).hexdigest()
            if proj.config else None
        )
        cache[proj.name] = {
            "source_hash": current_hash,
            "source_layers": source_layers,
            "artifact_count": len(all_artifacts),
            "config_hash": config_hash,
        }

    _save_projection_cache(build_dir, cache)
    return stats


def _materialize_projection(
    proj: Projection,
    layer_artifacts: dict[str, list[Artifact]],
    build_dir: Path,
    source_override: list[dict] | None = None,
    logger: SynixLogger | None = None,
    triggered_by: str | None = None,
) -> bool:
    """Materialize a single projection.

    Args:
        source_override: If provided, use these sources instead of proj.sources.
            Used for progressive (intermediate) materialization.
        logger: Optional logger for projection events.
        triggered_by: Name of the layer that triggered this materialization.

    Returns:
        True if materialization was skipped (no-op), False if work was done.
    """
    sources = source_override if source_override is not None else proj.sources

    all_artifacts: list[Artifact] = []
    sources_config: list[dict] = []

    for source in sources:
        layer_name = source["layer"]
        artifacts = layer_artifacts.get(layer_name, [])
        all_artifacts.extend(artifacts)
        for art in artifacts:
            level = art.metadata.get("layer_level", 0)
            sources_config.append({"layer": layer_name, "level": level})
            break

    if logger:
        logger.projection_start(proj.name, proj.projection_type,
                                triggered_by=triggered_by)

    if proj.projection_type == "search_index":
        # Use projection registry to avoid direct search import
        try:
            projection = get_projection("search_index", build_dir)
        except ValueError:
            # Search module not loaded — skip silently
            return True
        config = dict(proj.config)  # preserve projection-level config (embedding_config, etc.)
        config["sources"] = sources_config
        if logger:
            config["_synix_logger"] = logger
        projection.materialize(all_artifacts, config)
        projection.close()
    elif proj.projection_type == "flat_file":
        projection = FlatFileProjection()
        config = dict(proj.config)
        if "output_path" not in config:
            config["output_path"] = str(build_dir / "context.md")
        projection.materialize(all_artifacts, config)

    if logger:
        logger.projection_finish(proj.name, triggered_by=triggered_by)

    return False


PROJECTION_CACHE_FILE = ".projection_cache.json"


def _compute_projection_hash(artifacts: list[Artifact], proj_config: dict | None = None) -> str:
    """Compute a hash over sorted artifact content hashes and projection config."""
    hashes = sorted(a.content_hash for a in artifacts if a.content_hash)
    parts = "|".join(hashes)
    if proj_config:
        config_str = json.dumps(proj_config, sort_keys=True, default=str)
        parts += "|config:" + config_str
    return hashlib.sha256(parts.encode()).hexdigest()


def _load_projection_cache(build_dir: Path) -> dict:
    """Load projection cache from build_dir/.projection_cache.json."""
    cache_path = build_dir / PROJECTION_CACHE_FILE
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_projection_cache(build_dir: Path, cache: dict) -> None:
    """Save projection cache to build_dir/.projection_cache.json."""
    cache_path = build_dir / PROJECTION_CACHE_FILE
    cache_path.write_text(json.dumps(cache, indent=2))
