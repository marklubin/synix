"""Pipeline runner — walk DAG, run transforms, cache artifacts."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from synix.build.artifacts import ArtifactStore
from synix.build.dag import compute_levels, needs_rebuild, resolve_build_order
from synix.build.fingerprint import Fingerprint, compute_build_fingerprint
from synix.build.projections import FlatFileProjection, get_projection
from synix.build.provenance import ProvenanceTracker
from synix.build.search_surfaces import (
    search_surface_ready,
    surface_local_path,
    transform_runtime_search_updates,
    validate_search_surface_uses,
)
from synix.build.snapshots import BuildTransaction, commit_build_snapshot, start_build_transaction
from synix.core.logging import SynixLogger, Verbosity
from synix.core.models import (
    Artifact,
    FlatFile,
    Layer,
    Pipeline,
    SearchIndex,
    SearchSurface,
    Source,
    Transform,
    TransformContext,
)

logger = logging.getLogger(__name__)


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
    snapshot_oid: str | None = None
    manifest_oid: str | None = None
    head_ref: str | None = None
    run_ref: str | None = None
    synix_dir: str | None = None


def run(
    pipeline: Pipeline,
    source_dir: str | None = None,
    verbosity: int = 0,
    concurrency: int = 5,
    progress=None,
    validate: bool = False,
) -> RunResult:
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

    # Compute levels from DAG structure
    compute_levels(pipeline.layers)
    validate_search_surface_uses(pipeline)

    # Create structured logger
    slogger = SynixLogger(
        verbosity=Verbosity(min(verbosity, Verbosity.DEBUG)),
        build_dir=build_dir,
        progress=progress,
    )
    snapshot_txn = start_build_transaction(pipeline, build_dir, slogger.run_log.run_id)

    # Resolve build order
    build_order = resolve_build_order(pipeline)
    slogger.run_start(pipeline.name, len(build_order))

    # Artifacts produced per layer (layer_name -> list[Artifact])
    layer_artifacts: dict[str, list[Artifact]] = {}

    for layer in build_order:
        layer_start = time.time()
        stats = LayerStats(name=layer.name, level=layer._level)
        slogger.layer_start(layer.name, layer._level)

        if isinstance(layer, Source):
            # Source layer — call load()
            source_config = _build_source_config(pipeline, layer, src_dir)
            source_config["_logger"] = slogger
            source_config["_layer_name"] = layer.name

            try:
                artifacts = layer.load(source_config)
            except Exception:
                logger.warning("Source %s failed to load", layer.name, exc_info=True)
                artifacts = []

            # Save source artifacts
            for artifact in artifacts:
                artifact.metadata["layer_name"] = layer.name
                artifact.metadata["layer_level"] = layer._level
                store.save_artifact(artifact, layer.name, layer._level)
                provenance.record(artifact.label, parent_labels=[], prompt_id=None, model_config=None)
                _record_snapshot_artifact(
                    snapshot_txn,
                    artifact,
                    layer_name=layer.name,
                    layer_level=layer._level,
                    parent_labels=[],
                )
                stats.built += 1
                slogger.artifact_built(layer.name, artifact.label)

            layer_artifacts[layer.name] = artifacts

        elif isinstance(layer, Transform):
            # Transform layer — gather inputs, split, execute
            inputs = _gather_inputs(layer, layer_artifacts, store)
            transform_config = _build_transform_config(pipeline, layer, src_dir, build_dir)
            transform_ctx = _build_transform_context(pipeline, layer, src_dir, build_dir, transform_config)

            # Compute transform fingerprint
            transform_fp = layer.compute_fingerprint(transform_config)

            layer_built: list[Artifact] = []

            if _layer_fully_cached(layer, inputs, store, transform_fp):
                # All cached — load existing artifacts
                existing = store.list_artifacts(layer.name)
                for art in existing:
                    art.metadata["layer_name"] = layer.name
                    art.metadata["layer_level"] = layer._level
                    layer_built.append(art)
                    _record_snapshot_artifact(
                        snapshot_txn,
                        art,
                        layer_name=layer.name,
                        layer_level=layer._level,
                        parent_labels=_snapshot_parent_labels(art, inputs, provenance),
                    )
                    stats.cached += 1
                    slogger.artifact_cached(layer.name, art.label)
            else:
                # Execute transform
                transform_ctx = transform_ctx.with_updates({"_logger": slogger, "_layer_name": layer.name})

                def _save_artifact(
                    artifact: Artifact,
                    *,
                    parent_inputs: list[Artifact] | None = None,
                    _layer=layer,
                    _transform_fp=transform_fp,
                    _inputs=inputs,
                ) -> None:
                    effective_inputs = parent_inputs if parent_inputs is not None else _inputs
                    # Compute per-artifact build fingerprint
                    build_fp = compute_build_fingerprint(_transform_fp, artifact.input_ids)

                    rebuild, _reasons = needs_rebuild(
                        artifact.label,
                        artifact.input_ids,
                        store,
                        current_build_fingerprint=build_fp,
                    )
                    if rebuild:
                        artifact.metadata["layer_name"] = _layer.name
                        artifact.metadata["layer_level"] = _layer._level
                        artifact.metadata["build_fingerprint"] = build_fp.to_dict()
                        artifact.metadata["transform_fingerprint"] = _transform_fp.to_dict()
                        store.save_artifact(artifact, _layer.name, _layer._level)
                        parent_labels = _provenance_parent_labels(artifact, effective_inputs, provenance)
                        provenance.record(
                            artifact.label,
                            parent_labels=parent_labels,
                            prompt_id=artifact.prompt_id,
                            model_config=artifact.model_config,
                        )
                        _record_snapshot_artifact(
                            snapshot_txn,
                            artifact,
                            layer_name=_layer.name,
                            layer_level=_layer._level,
                            parent_labels=parent_labels,
                        )
                        layer_built.append(artifact)
                        stats.built += 1
                        slogger.artifact_built(_layer.name, artifact.label)
                    else:
                        cached = store.load_artifact(artifact.label)
                        if cached is not None:
                            cached.metadata["layer_name"] = _layer.name
                            cached.metadata["layer_level"] = _layer._level
                            layer_built.append(cached)
                            _record_snapshot_artifact(
                                snapshot_txn,
                                cached,
                                layer_name=_layer.name,
                                layer_level=_layer._level,
                                parent_labels=_snapshot_parent_labels(cached, effective_inputs, provenance),
                            )
                        else:
                            layer_built.append(artifact)
                            _record_snapshot_artifact(
                                snapshot_txn,
                                artifact,
                                layer_name=_layer.name,
                                layer_level=_layer._level,
                                parent_labels=_snapshot_parent_labels(artifact, effective_inputs, provenance),
                            )
                        stats.cached += 1
                        slogger.artifact_cached(_layer.name, artifact.label)

                def _on_batch_complete(artifacts: list[Artifact], unit_inputs: list[Artifact]) -> None:
                    for artifact in artifacts:
                        _save_artifact(artifact, parent_inputs=unit_inputs)

                # Build lookup of cached artifacts by sorted input_ids for per-unit cache checks
                existing_artifacts = store.list_artifacts(layer.name)
                cached_by_inputs: dict[tuple[str, ...], list[Artifact]] = {}
                for art in existing_artifacts:
                    stored_tfp_data = art.metadata.get("transform_fingerprint")
                    if stored_tfp_data is not None:
                        stored_fp = Fingerprint.from_dict(stored_tfp_data)
                        if transform_fp.matches(stored_fp):
                            key = tuple(sorted(art.input_ids))
                            cached_by_inputs.setdefault(key, []).append(art)

                def _on_cached(cached_arts: list[Artifact], unit_inputs: list[Artifact]) -> None:
                    for cached_art in cached_arts:
                        cached_art.metadata["layer_name"] = layer.name
                        cached_art.metadata["layer_level"] = layer._level
                        layer_built.append(cached_art)
                        _record_snapshot_artifact(
                            snapshot_txn,
                            cached_art,
                            layer_name=layer.name,
                            layer_level=layer._level,
                            parent_labels=_snapshot_parent_labels(cached_art, unit_inputs, provenance),
                        )
                        stats.cached += 1
                        slogger.artifact_cached(layer.name, cached_art.label)

                # Split inputs into work units
                units = layer.split(inputs, transform_ctx)
                use_concurrent = concurrency > 1 and len(units) > 1

                if use_concurrent:
                    _execute_transform_concurrent(
                        layer,
                        units,
                        transform_ctx,
                        concurrency,
                        on_complete=_on_batch_complete,
                        cached_by_inputs=cached_by_inputs,
                        on_cached=_on_cached,
                    )
                else:
                    for unit_inputs, config_extras in units:
                        # Per-unit cache check
                        unit_input_ids = tuple(sorted(a.artifact_id for a in unit_inputs if a.artifact_id))
                        cached_arts = cached_by_inputs.get(unit_input_ids)
                        if cached_arts:
                            _on_cached(cached_arts, unit_inputs)
                            continue
                        unit_ctx = transform_ctx.with_updates(config_extras)
                        new_artifacts = layer.execute(unit_inputs, unit_ctx)
                        for artifact in new_artifacts:
                            _save_artifact(artifact, parent_inputs=unit_inputs)

            layer_artifacts[layer.name] = layer_built

        stats.time_seconds = time.time() - layer_start
        result.layer_stats.append(stats)
        result.built += stats.built
        result.cached += stats.cached
        result.skipped += stats.skipped
        slogger.layer_finish(layer.name, stats.built, stats.cached)

        _materialize_layer_search_surfaces(pipeline, layer.name, layer_artifacts, store, build_dir, logger=slogger)

        # Materialize intermediate projections for this layer
        _materialize_layer_projections(pipeline, layer.name, layer_artifacts, store, build_dir, logger=slogger)

    # Materialize all final projections (with caching)
    result.projection_stats = _materialize_all_projections(pipeline, layer_artifacts, store, build_dir, logger=slogger)
    _refresh_surface_cache(pipeline, layer_artifacts, build_dir)

    # Run domain validators if requested and declared
    if validate and pipeline.validators:
        from synix.build.validators import run_validators

        result.validation = run_validators(pipeline, store, provenance)

    # Non-validating builds still record a snapshot; validating builds only
    # advance snapshot refs when all validators pass.
    if result.validation is None or result.validation.passed:
        snapshot_txn.assert_complete(layer_artifacts)
        snapshot_info = commit_build_snapshot(snapshot_txn)
        result.snapshot_oid = snapshot_info["snapshot_oid"]
        result.manifest_oid = snapshot_info["manifest_oid"]
        result.head_ref = snapshot_info["head_ref"]
        result.run_ref = snapshot_info["run_ref"]
        result.synix_dir = snapshot_info["synix_dir"]

    result.total_time = time.time() - start_time
    slogger.run_finish(result.total_time)
    result.run_log = slogger.run_log.to_dict()
    return result


def _build_source_config(pipeline: Pipeline, source: Source, src_dir: str) -> dict:
    """Build config dict for a Source layer."""
    config: dict = {"source_dir": src_dir}
    if source.dir:
        config["source_dir"] = source.dir
    return config


def _record_snapshot_artifact(
    snapshot_txn: BuildTransaction,
    artifact: Artifact,
    *,
    layer_name: str,
    layer_level: int,
    parent_labels: list[str],
) -> None:
    """Record the canonical artifact state used by this run into the snapshot transaction."""
    snapshot_txn.record_artifact(
        artifact,
        layer_name=layer_name,
        layer_level=layer_level,
        parent_labels=parent_labels,
    )


def _snapshot_parent_labels(
    artifact: Artifact,
    inputs: list[Artifact],
    provenance: ProvenanceTracker,
) -> list[str]:
    """Parent labels for immutable snapshots.

    Snapshots should not invent broad parent sets when an artifact explicitly
    declared inputs but they could not be resolved unambiguously.
    """
    derived = _get_parent_labels(artifact, inputs)
    if derived:
        return derived
    stored = provenance.get_parents(artifact.label)
    if stored:
        return stored
    if artifact.input_ids:
        return []
    return [inp.label for inp in inputs]


def _provenance_parent_labels(
    artifact: Artifact,
    inputs: list[Artifact],
    provenance: ProvenanceTracker,
) -> list[str]:
    """Parent labels for the legacy provenance surface.

    Provenance records keep the pre-snapshot best-effort behavior so existing
    lineage commands remain informative even when transforms omit explicit
    input_ids for aggregate artifacts.
    """
    derived = _get_parent_labels(artifact, inputs)
    if derived:
        return derived
    stored = provenance.get_parents(artifact.label)
    if stored:
        return stored
    return [inp.label for inp in inputs]


def _build_transform_config(pipeline: Pipeline, layer: Transform, src_dir: str, build_dir: Path) -> dict:
    """Build fingerprint-stable config dict for a Transform layer."""
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
    return transform_config


def _build_transform_context(
    pipeline: Pipeline,
    layer: Transform,
    src_dir: str,
    build_dir: Path,
    transform_config: dict,
) -> TransformContext:
    """Build the runtime transform context with explicit capabilities."""
    runtime_updates = {
        "workspace": {
            "source_dir": src_dir,
            "build_dir": str(build_dir),
        }
    }
    runtime_updates.update(
        transform_runtime_search_updates(
            layer,
            build_dir=build_dir,
            projections=pipeline.projections,
        )
    )
    return TransformContext.from_value(transform_config).with_updates(runtime_updates)


def _materialize_layer_search_surfaces(
    pipeline: Pipeline,
    layer_name: str,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
    build_dir: Path,
    logger: SynixLogger | None = None,
) -> None:
    """Materialize search surfaces that source from this layer.

    Search surfaces only materialize once their full source closure is
    available, so the on-disk compatibility DB always matches cache state.
    """
    for surface in pipeline.surfaces:
        source_layer_names = [s.name for s in surface.sources]
        if layer_name not in source_layer_names:
            continue

        available_names = set(layer_artifacts)
        if not search_surface_ready(surface, available_names):
            continue

        _materialize_projection(
            surface,
            layer_artifacts,
            build_dir,
            logger=logger,
            triggered_by=layer_name,
        )


def _gather_inputs(
    layer: Layer,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
) -> list[Artifact]:
    """Gather inputs from dependent layers."""
    inputs: list[Artifact] = []
    for dep in layer.depends_on:
        dep_artifacts = layer_artifacts.get(dep.name)
        if dep_artifacts is None:
            # Load from store if not in memory (e.g., cached from previous run)
            dep_artifacts = store.list_artifacts(dep.name)
        inputs.extend(dep_artifacts)
    return inputs


def _layer_fully_cached(
    layer: Layer,
    inputs: list[Artifact],
    store: ArtifactStore,
    transform_fp: Fingerprint | None = None,
) -> bool:
    """Check if a layer can be entirely skipped (all artifacts cached)."""
    existing = store.list_artifacts(layer.name)
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


def _get_parent_labels(artifact: Artifact, inputs: list[Artifact]) -> list[str]:
    """Determine parent labels based on input IDs (hashes)."""
    hash_to_labels: dict[str, list[str]] = {}
    for inp in inputs:
        if inp.artifact_id:
            hash_to_labels.setdefault(inp.artifact_id, []).append(inp.label)

    parents = []
    for h in artifact.input_ids:
        labels = hash_to_labels.get(h, [])
        if len(labels) == 1:
            parents.append(labels[0])
        elif len(labels) > 1:
            return []

    if not parents and inputs:
        parents = [inp.label for inp in inputs]

    return parents


def _execute_transform_concurrent(
    transform: Transform,
    units: list[tuple[list[Artifact], dict]],
    config: TransformContext | dict,
    concurrency: int,
    on_complete=None,
    cached_by_inputs: dict[tuple[str, ...], list[Artifact]] | None = None,
    on_cached=None,
) -> list[Artifact]:
    """Execute transform work units concurrently.

    Each unit is (unit_inputs, config_extras) as returned by transform.split().
    Uses copy.copy(transform) per worker as a best-effort guard against
    accidental self mutation (shallow copy only — does NOT isolate nested
    mutable state).

    Units whose input_ids match ``cached_by_inputs`` are skipped (reported
    via ``on_cached``) and never submitted to the thread pool.
    """

    def _invoke_callback(callback, artifacts: list[Artifact], unit_inputs: list[Artifact]) -> None:
        if callback is None:
            return
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            callback(artifacts, unit_inputs)
            return

        params = list(signature.parameters.values())
        if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
            callback(artifacts, unit_inputs)
            return

        positional = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            callback(artifacts, unit_inputs)
        elif len(positional) == 1:
            callback(artifacts)
        else:
            callback()

    # Filter out cached units before submitting to pool
    units_to_run: list[tuple[int, list[Artifact], dict]] = []
    for i, (unit_inputs, config_extras) in enumerate(units):
        if cached_by_inputs:
            unit_input_ids = tuple(sorted(a.artifact_id for a in unit_inputs if a.artifact_id))
            cached_arts = cached_by_inputs.get(unit_input_ids)
            if cached_arts:
                _invoke_callback(on_cached, cached_arts, unit_inputs)
                continue
        units_to_run.append((i, unit_inputs, config_extras))

    if not units_to_run:
        return []

    results: list[list[Artifact] | Exception] = [None] * len(units_to_run)  # type: ignore[list-item]

    # Keys that should be shared (not deep-copied) across workers because they
    # contain non-picklable objects (e.g. logger with file handles).
    base_ctx = TransformContext.from_value(config)
    base_config = base_ctx.to_dict()
    _shared_keys = {"_logger"}

    # Pre-create a shared LLM client to avoid per-thread connection overhead
    shared_client = None
    try:
        from synix.build.llm_transforms import _make_llm_client

        shared_client = _make_llm_client(base_config)
    except ImportError:
        pass  # Non-LLM transforms don't need a shared client
    except Exception:
        logger.warning("Could not create shared LLM client; workers will create their own", exc_info=True)

    def _run_one(index: int, unit_inputs: list[Artifact], config_extras: dict) -> tuple[int, list[Artifact]]:
        """Execute the transform for a single unit, returning (index, artifacts)."""
        # copy.copy() per worker: best-effort guard (shallow only)
        worker_transform = copy.copy(transform)
        # Deep-copy mutable config to avoid cross-thread mutation, but share
        # non-copyable objects like the logger.
        shared = {k: base_config[k] for k in _shared_keys if k in base_config}
        copyable = {k: v for k, v in base_config.items() if k not in _shared_keys}
        worker_config = copy.deepcopy(copyable)
        worker_config.update(shared)
        worker_config.update(config_extras)
        if shared_client is not None:
            worker_config["_shared_llm_client"] = shared_client
        worker_ctx = TransformContext.from_value(worker_config)
        return index, worker_transform.execute(unit_inputs, worker_ctx)

    first_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, seq_idx, unit_inputs, config_extras): seq_idx
            for seq_idx, (_orig_idx, unit_inputs, config_extras) in enumerate(units_to_run)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, artifacts = future.result()
                results[idx] = artifacts
                _orig_idx, unit_inputs, _config_extras = units_to_run[idx]
                _invoke_callback(on_complete, artifacts, unit_inputs)
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
        source_layer_names = [s.name for s in proj.sources]
        if layer_name not in source_layer_names:
            continue

        if isinstance(proj, FlatFile):
            # Flat file only makes sense with all sources (e.g., core layer)
            if not all(ln in layer_artifacts for ln in source_layer_names):
                continue
            _materialize_projection(proj, layer_artifacts, build_dir, logger=logger, triggered_by=layer_name)
        elif isinstance(proj, SearchIndex):
            # Progressive: materialize with whatever sources are available
            available_names = [ln for ln in source_layer_names if ln in layer_artifacts]
            if available_names:
                _materialize_projection(
                    proj,
                    layer_artifacts,
                    build_dir,
                    source_layer_override=available_names,
                    logger=logger,
                    triggered_by=layer_name,
                )
        else:
            # Unknown projection type — require all sources
            if not all(ln in layer_artifacts for ln in source_layer_names):
                continue
            _materialize_projection(proj, layer_artifacts, build_dir, logger=logger, triggered_by=layer_name)


def _materialize_all_projections(
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
    build_dir: Path,
    logger: SynixLogger | None = None,
) -> list[ProjectionStats]:
    """Materialize all projections after the full build (with caching).

    Uses split hashing: content hash (artifact IDs) and embedding config
    hash are tracked independently. When only the embedding config changes,
    FTS5 is preserved and only embeddings are regenerated.
    """
    cache = _load_projection_cache(build_dir)
    stats: list[ProjectionStats] = []

    # Determine the last source layer for each projection (for triggered_by)
    build_order = resolve_build_order(pipeline)
    layer_order = {layer.name: i for i, layer in enumerate(build_order)}

    for proj in pipeline.projections:
        # Gather all artifacts for this projection
        all_artifacts: list[Artifact] = []
        source_layer_names = [s.name for s in proj.sources]
        for layer_name in source_layer_names:
            all_artifacts.extend(layer_artifacts.get(layer_name, []))

        # Determine the last source layer (by build order) for triggered_by
        last_layer = max(source_layer_names, key=lambda ln: layer_order.get(ln, 0)) if source_layer_names else None

        # Compute separate hashes for content and embedding config
        content_hash = _compute_content_only_hash(all_artifacts)
        embedding_hash = _compute_embedding_config_hash(proj) if isinstance(proj, SearchIndex) else None

        cached_entry = cache.get(proj.name)
        content_cached = (
            cached_entry is not None
            and cached_entry.get("content_hash") == content_hash
            and cached_entry.get("artifact_count") == len(all_artifacts)
        )
        embedding_cached = cached_entry is not None and cached_entry.get("embedding_hash") == embedding_hash

        if content_cached and embedding_cached:
            # Fully cached — skip everything
            if logger:
                logger.projection_cached(proj.name, triggered_by=last_layer)
            stats.append(ProjectionStats(name=proj.name, status="cached"))
        elif content_cached and not embedding_cached and isinstance(proj, SearchIndex):
            # FTS5 is fine, only re-embed
            if logger:
                logger.projection_start(proj.name, "search_index", triggered_by=last_layer)
            _regenerate_embeddings_only(proj, all_artifacts, build_dir, logger=logger)
            if logger:
                logger.projection_finish(proj.name, triggered_by=last_layer)
            stats.append(ProjectionStats(name=proj.name, status="built"))
        else:
            # Full rebuild
            cached = _materialize_projection(
                proj,
                layer_artifacts,
                build_dir,
                logger=logger,
                triggered_by=last_layer,
            )
            status = "cached" if cached else "built"
            stats.append(ProjectionStats(name=proj.name, status=status))

        # Update cache with split hashes
        cache[proj.name] = {
            "content_hash": content_hash,
            "embedding_hash": embedding_hash,
            "source_layers": source_layer_names,
            "artifact_count": len(all_artifacts),
        }

    _save_projection_cache(build_dir, cache)
    return stats


def _refresh_surface_cache(
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
    build_dir: Path,
) -> None:
    """Persist final cache metadata for build-time search surfaces."""
    cache = _load_projection_cache(build_dir)

    for surface in pipeline.surfaces:
        all_artifacts: list[Artifact] = []
        source_layer_names = [s.name for s in surface.sources]
        for layer_name in source_layer_names:
            all_artifacts.extend(layer_artifacts.get(layer_name, []))

        cache[surface.name] = {
            "content_hash": _compute_content_only_hash(all_artifacts),
            "embedding_hash": _compute_embedding_config_hash(surface),
            "source_layers": source_layer_names,
            "artifact_count": len(all_artifacts),
        }

    _save_projection_cache(build_dir, cache)


def _get_projection_config(proj: Layer) -> dict:
    """Extract cache-relevant config from a projection or search surface."""
    if isinstance(proj, SearchIndex):
        return {
            "search": proj.search,
            "embedding_config": proj.embedding_config,
        }
    elif isinstance(proj, SearchSurface):
        return {
            "modes": proj.modes,
            "embedding_config": proj.embedding_config,
        }
    elif isinstance(proj, FlatFile):
        return {"output_path": proj.output_path}
    return proj.config


def _materialize_projection(
    proj: Layer,
    layer_artifacts: dict[str, list[Artifact]],
    build_dir: Path,
    source_layer_override: list[str] | None = None,
    logger: SynixLogger | None = None,
    triggered_by: str | None = None,
) -> bool:
    """Materialize a single projection.

    Args:
        source_layer_override: If provided, use these layer names instead of proj.sources.
            Used for progressive (intermediate) materialization.
        logger: Optional logger for projection events.
        triggered_by: Name of the layer that triggered this materialization.

    Returns:
        True if materialization was skipped (no-op), False if work was done.
    """
    source_layer_names = source_layer_override if source_layer_override is not None else [s.name for s in proj.sources]

    all_artifacts: list[Artifact] = []
    sources_config: list[dict] = []

    for layer_name in source_layer_names:
        artifacts = layer_artifacts.get(layer_name, [])
        all_artifacts.extend(artifacts)
        for art in artifacts:
            level = art.metadata.get("layer_level", 0)
            sources_config.append({"layer": layer_name, "level": level})
            break

    if isinstance(proj, SearchIndex):
        proj_type = "search_index"
    elif isinstance(proj, SearchSurface):
        proj_type = "search_surface"
    elif isinstance(proj, FlatFile):
        proj_type = "flat_file"
    else:
        proj_type = "unknown"

    if logger:
        logger.projection_start(proj.name, proj_type, triggered_by=triggered_by)

    if isinstance(proj, (SearchIndex, SearchSurface)):
        # Use projection registry to avoid direct search import
        try:
            db_path = build_dir / "search.db" if isinstance(proj, SearchIndex) else surface_local_path(build_dir, proj)
            projection = get_projection("search_index", build_dir, db_path)
        except ValueError as exc:
            logger.warning("Projection %r unavailable: %s", proj.name, exc)
            return True
        config = dict(proj.config)
        config["embedding_config"] = proj.embedding_config
        config["sources"] = sources_config
        if logger:
            config["_synix_logger"] = logger
        projection.materialize(all_artifacts, config)
        projection.close()
    elif isinstance(proj, FlatFile):
        projection = FlatFileProjection()
        config = {"output_path": proj.output_path}
        projection.materialize(all_artifacts, config)

    if logger:
        logger.projection_finish(proj.name, triggered_by=triggered_by)

    return False


PROJECTION_CACHE_FILE = ".projection_cache.json"


def _compute_projection_hash(artifacts: list[Artifact], proj_config: dict | None = None) -> str:
    """Compute a hash over sorted artifact IDs (hashes) and projection config."""
    hashes = sorted(a.artifact_id for a in artifacts if a.artifact_id)
    parts = "|".join(hashes)
    if proj_config:
        config_str = json.dumps(proj_config, sort_keys=True, default=str)
        parts += "|config:" + config_str
    return hashlib.sha256(parts.encode()).hexdigest()


def _compute_content_only_hash(artifacts: list[Artifact]) -> str:
    """Compute a hash over sorted artifact IDs only (no config)."""
    hashes = sorted(a.artifact_id for a in artifacts if a.artifact_id)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _compute_embedding_config_hash(proj) -> str | None:
    """Compute a hash over embedding identity fields only."""
    if not isinstance(proj, (SearchIndex, SearchSurface)):
        return None
    if not proj.embedding_config:
        return None
    # Only identity fields: provider, model, dimensions
    identity = {
        "provider": proj.embedding_config.get("provider", "fastembed"),
        "model": proj.embedding_config.get("model", "BAAI/bge-small-en-v1.5"),
        "dimensions": proj.embedding_config.get("dimensions"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _regenerate_embeddings_only(
    proj,
    artifacts: list[Artifact],
    build_dir: Path,
    logger=None,
) -> None:
    """Re-generate embeddings without rebuilding FTS5."""
    embedding_config = proj.embedding_config
    if not embedding_config or not artifacts:
        return
    try:
        projection = get_projection("search_index", build_dir)
    except ValueError:
        logger.warning("Cannot regenerate embeddings: search_index projection unavailable")
        return
    synix_logger = logger if logger else None
    projection._generate_embeddings(embedding_config, artifacts, synix_logger)


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
