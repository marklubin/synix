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

from synix.build.dag import compute_levels, needs_rebuild, resolve_build_order
from synix.build.error_classifier import (
    DeadLetterQueue,
    ErrorClassifier,
    ErrorVerdict,
)
from synix.build.fingerprint import Fingerprint, compute_build_fingerprint
from synix.build.projections import get_projection
from synix.build.refs import synix_dir_for_build_dir
from synix.build.search_surfaces import (
    search_surface_ready,
    surface_local_path,
    transform_runtime_search_updates,
    validate_search_surface_uses,
)
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.build.snapshots import (
    BuildTransaction,
    clear_checkpoints,
    commit_build_snapshot,
    start_build_transaction,
    write_layer_checkpoint,
)
from synix.core.logging import SynixLogger, Verbosity
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

logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """Build statistics for a single layer."""

    name: str
    level: int
    built: int = 0
    cached: int = 0
    skipped: int = 0
    dlq_count: int = 0
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
    dlq: DeadLetterQueue = field(default_factory=DeadLetterQueue)
    validation: object | None = None  # ValidationResult when validators are declared
    snapshot_oid: str | None = None
    manifest_oid: str | None = None
    head_ref: str | None = None
    run_ref: str | None = None
    synix_dir: str | None = None


def _surface_db_path(surface: SearchSurface, work_dir: Path) -> Path:
    """Return the local DB path for a build-time search surface."""
    return surface_local_path(work_dir, surface)


def run(
    pipeline: Pipeline,
    source_dir: str | None = None,
    verbosity: int = 0,
    concurrency: int = 5,
    progress=None,
    validate: bool = False,
    error_classifier: ErrorClassifier | None = None,
) -> RunResult:
    """Execute the full pipeline — walk DAG, run transforms, materialize projections.

    Args:
        pipeline: The pipeline definition.
        source_dir: Override for pipeline.source_dir.
        verbosity: Verbosity level (0=default, 1=verbose, 2=debug).
        concurrency: Number of concurrent LLM requests (1 = sequential).
        progress: Optional BuildProgress for live display updates.
        validate: Run domain validators after build (default False).
        error_classifier: Optional classifier for non-fatal errors. When None
            (default), all LLM errors are fatal and abort the build. Pass
            ``LLMErrorClassifier()`` to enable DLQ — content-filter and
            input-too-large errors will be skipped instead of aborting.

    Returns:
        RunResult with build statistics.
    """
    start_time = time.time()
    src_dir = source_dir or pipeline.source_dir
    build_dir = Path(pipeline.build_dir)
    synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=pipeline.synix_dir)
    synix_dir.mkdir(parents=True, exist_ok=True)
    work_dir = synix_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    result = RunResult()

    # Compute levels from DAG structure
    compute_levels(pipeline.layers)
    validate_search_surface_uses(pipeline)

    # Create structured logger
    slogger = SynixLogger(
        verbosity=Verbosity(min(verbosity, Verbosity.DEBUG)),
        build_dir=synix_dir,
        progress=progress,
    )
    result.dlq.slogger = slogger
    snapshot_txn = start_build_transaction(pipeline, build_dir, slogger.run_log.run_id)
    store = SnapshotArtifactCache(snapshot_txn.synix_dir)

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

            # Record source artifacts in snapshot
            for artifact in artifacts:
                artifact.metadata["layer_name"] = layer.name
                artifact.metadata["layer_level"] = layer._level
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
            transform_config = _build_transform_config(pipeline, layer, src_dir, work_dir)
            transform_ctx = _build_transform_context(pipeline, layer, src_dir, work_dir, transform_config)

            # Compute transform fingerprint
            transform_fp = layer.compute_fingerprint(transform_config)

            layer_built: list[Artifact] = []
            dlq_before = len(result.dlq)

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
                        parent_labels=_snapshot_parent_labels(art, inputs, store),
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
                        parent_labels = _get_parent_labels(artifact, effective_inputs)
                        if not parent_labels:
                            parent_labels = [inp.label for inp in effective_inputs]
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
                                parent_labels=_snapshot_parent_labels(cached, effective_inputs, store),
                            )
                        else:
                            layer_built.append(artifact)
                            _record_snapshot_artifact(
                                snapshot_txn,
                                artifact,
                                layer_name=_layer.name,
                                layer_level=_layer._level,
                                parent_labels=_snapshot_parent_labels(artifact, effective_inputs, store),
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
                            parent_labels=_snapshot_parent_labels(cached_art, unit_inputs, store),
                        )
                        stats.cached += 1
                        slogger.artifact_cached(layer.name, cached_art.label)

                # Split inputs into work units
                units = _invoke_transform_split(layer, inputs, transform_ctx)
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
                        dlq=result.dlq if error_classifier else None,
                        error_classifier=error_classifier,
                        layer_name=layer.name,
                    )
                else:
                    for unit_inputs, config_extras in units:
                        # Per-unit cache check
                        unit_input_ids = tuple(sorted(a.artifact_id for a in unit_inputs if a.artifact_id))
                        cached_arts = cached_by_inputs.get(unit_input_ids)
                        if cached_arts:
                            _on_cached(cached_arts, unit_inputs)
                            continue
                        try:
                            unit_ctx = transform_ctx.with_updates(config_extras)
                            new_artifacts = _invoke_transform_execute(layer, unit_inputs, unit_ctx)
                            for artifact in new_artifacts:
                                _save_artifact(artifact, parent_inputs=unit_inputs)
                        except Exception as exc:
                            if error_classifier is None:
                                raise
                            labels = [a.label for a in unit_inputs[:3]]
                            artifact_desc = ", ".join(labels)
                            verdict = error_classifier.classify(exc, artifact_desc)
                            if verdict == ErrorVerdict.DLQ:
                                result.dlq.add(artifact_desc, exc, layer_name=layer.name)
                            else:
                                raise

            layer_artifacts[layer.name] = layer_built
            stats.dlq_count = len(result.dlq) - dlq_before

        stats.time_seconds = time.time() - layer_start
        result.layer_stats.append(stats)
        result.built += stats.built
        result.cached += stats.cached
        result.skipped += stats.skipped
        slogger.layer_finish(layer.name, stats.built, stats.cached)

        # Checkpoint after each layer so interrupted builds can recover
        write_layer_checkpoint(snapshot_txn, layer.name)

        _materialize_layer_search_surfaces(pipeline, layer.name, layer_artifacts, work_dir, logger=slogger)

    # Record projection declarations in the snapshot transaction
    _record_snapshot_projections(snapshot_txn, pipeline, layer_artifacts)

    # Refresh the in-memory cache with the current build's artifacts so
    # downstream consumers (validators, fixers) see newly built data.
    store.update_from_build(layer_artifacts)

    # Run domain validators if requested and declared
    if validate and pipeline.validators:
        from synix.build.validators import run_validators

        result.validation = run_validators(pipeline, store)

    # Non-validating builds still record a snapshot; validating builds only
    # advance snapshot refs when all validators pass.
    if result.validation is None or result.validation.passed:
        # Persist DLQ entries in the manifest so post-build inspection
        # can explain missing artifacts.
        if len(result.dlq) > 0:
            snapshot_txn.dlq_entries = [
                {
                    "artifact_desc": e.artifact_desc,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "layer_name": e.layer_name,
                }
                for e in result.dlq.entries
            ]
        snapshot_txn.assert_complete(layer_artifacts)
        snapshot_info = commit_build_snapshot(snapshot_txn)
        result.snapshot_oid = snapshot_info["snapshot_oid"]
        result.manifest_oid = snapshot_info["manifest_oid"]
        result.head_ref = snapshot_info["head_ref"]
        result.run_ref = snapshot_info["run_ref"]
        result.synix_dir = snapshot_info["synix_dir"]

        # Successful commit supersedes all prior checkpoints
        clear_checkpoints(snapshot_txn.synix_dir)

    result.total_time = time.time() - start_time
    if len(result.dlq) > 0:
        logger.warning("Build completed with DLQ entries: %s", result.dlq.summary())
        slogger.run_log.dlq_entries = snapshot_txn.dlq_entries
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


def _projection_config_fingerprint(config: dict) -> str:
    """Compute a sha256 fingerprint of canonical JSON projection config."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _record_snapshot_projections(
    snapshot_txn: BuildTransaction,
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
) -> None:
    """Record structured projection declarations into the snapshot transaction."""
    for proj in pipeline.projections:
        source_layer_names = [s.name for s in proj.sources]
        input_labels = sorted(a.label for layer_name in source_layer_names for a in layer_artifacts.get(layer_name, []))

        if isinstance(proj, SynixSearch):
            adapter = "synix_search"
            config: dict = {
                "modes": list(proj.search),
                "embedding_config": dict(proj.embedding_config) if proj.embedding_config else {},
            }
        elif isinstance(proj, SearchIndex):
            adapter = "synix_search"
            config = {
                "modes": list(proj.search),
                "embedding_config": dict(proj.embedding_config) if proj.embedding_config else {},
            }
        elif isinstance(proj, FlatFile):
            adapter = "flat_file"
            config = {
                "output_path": proj.output_path,
            }
        else:
            adapter = type(proj).__name__.lower()
            config = dict(proj.config) if proj.config else {}

        snapshot_txn.record_projection(
            proj.name,
            adapter=adapter,
            input_artifact_labels=input_labels,
            config=config,
            config_fingerprint=_projection_config_fingerprint(config),
        )


def _snapshot_parent_labels(
    artifact: Artifact,
    inputs: list[Artifact],
    cache: SnapshotArtifactCache,
) -> list[str]:
    """Parent labels for immutable snapshots.

    Snapshots should not invent broad parent sets when an artifact explicitly
    declared inputs but they could not be resolved unambiguously.
    """
    derived = _get_parent_labels(artifact, inputs)
    if derived:
        return derived
    stored = cache.get_parents(artifact.label)
    if stored:
        return stored
    if artifact.input_ids:
        return []
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


def _transform_prefers_legacy_config_dict(method) -> bool:
    """Return True when ``method`` still advertises the old ``config: dict`` API."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        return False

    config_param = positional[1]
    annotation = config_param.annotation
    annotation_text = ""
    if annotation is not inspect.Signature.empty:
        if isinstance(annotation, str):
            annotation_text = annotation
        else:
            annotation_text = (
                getattr(annotation, "__qualname__", "") or getattr(annotation, "__name__", "") or str(annotation)
            )

    if annotation is dict or annotation_text == "dict" or annotation_text.startswith("dict["):
        return True
    if "TransformContext" in annotation_text:
        return False

    return config_param.name == "config"


def _transform_runtime_arg(method, ctx: TransformContext) -> TransformContext | dict:
    """Adapt ``ctx`` for legacy transforms that still expect a raw config dict."""
    if _transform_prefers_legacy_config_dict(method):
        return ctx.to_dict()
    return ctx


def _invoke_transform_split(
    transform: Transform,
    inputs: list[Artifact],
    ctx: TransformContext,
) -> list[tuple[list[Artifact], dict]]:
    """Invoke ``split`` with either ``TransformContext`` or a legacy plain dict."""
    return transform.split(inputs, _transform_runtime_arg(transform.split, ctx))


def _invoke_transform_execute(
    transform: Transform,
    inputs: list[Artifact],
    ctx: TransformContext,
) -> list[Artifact]:
    """Invoke ``execute`` with either ``TransformContext`` or a legacy plain dict."""
    return transform.execute(inputs, _transform_runtime_arg(transform.execute, ctx))


def _materialize_layer_search_surfaces(
    pipeline: Pipeline,
    layer_name: str,
    layer_artifacts: dict[str, list[Artifact]],
    work_dir: Path,
    logger: SynixLogger | None = None,
) -> None:
    """Materialize search surfaces to .synix/work/ for build-time use.

    Search surfaces only materialize once their full source closure is
    available, so the on-disk DB always matches cache state.
    """
    for surface in pipeline.surfaces:
        source_layer_names = [s.name for s in surface.sources]
        if layer_name not in source_layer_names:
            continue

        available_names = set(layer_artifacts)
        if not search_surface_ready(surface, available_names):
            continue

        _materialize_search_surface(
            surface,
            layer_artifacts,
            work_dir,
            logger=logger,
            triggered_by=layer_name,
        )


def _gather_inputs(
    layer: Layer,
    layer_artifacts: dict[str, list[Artifact]],
    store: SnapshotArtifactCache,
) -> list[Artifact]:
    """Gather inputs from dependent layers."""
    inputs: list[Artifact] = []
    for dep in layer.depends_on:
        dep_artifacts = layer_artifacts.get(dep.name)
        if dep_artifacts is None:
            # Load from snapshot cache if not in memory (e.g., cached from previous run)
            dep_artifacts = store.list_artifacts(dep.name)
        inputs.extend(dep_artifacts)
    return inputs


def _layer_fully_cached(
    layer: Layer,
    inputs: list[Artifact],
    store: SnapshotArtifactCache,
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
    dlq: DeadLetterQueue | None = None,
    error_classifier: ErrorClassifier | None = None,
    layer_name: str = "",
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
        return index, _invoke_transform_execute(worker_transform, unit_inputs, worker_ctx)

    first_fatal: Exception | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, seq_idx, unit_inputs, config_extras): seq_idx
            for seq_idx, (_orig_idx, unit_inputs, config_extras) in enumerate(units_to_run)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, artifacts = future.result(timeout=600)  # 10 min hard ceiling per work unit
                results[idx] = artifacts
                _orig_idx, unit_inputs, _config_extras = units_to_run[idx]
                _invoke_callback(on_complete, artifacts, unit_inputs)
            except TimeoutError:
                _orig_idx, unit_inputs, _config_extras = units_to_run[idx]
                labels = [a.label for a in unit_inputs[:3]]
                err = RuntimeError(f"Transform work unit timed out after 600s (inputs: {labels})")
                results[idx] = err
                if first_fatal is None:
                    first_fatal = err
                logger.warning("Work unit %d timed out (inputs: %s)", idx, labels)
            except Exception as exc:
                _orig_idx, unit_inputs, _config_extras = units_to_run[idx]
                labels = [a.label for a in unit_inputs[:3]]
                artifact_desc = ", ".join(labels)
                if error_classifier is None:
                    # No classifier → fail hard immediately
                    results[idx] = exc
                    if first_fatal is None:
                        first_fatal = exc
                else:
                    verdict = error_classifier.classify(exc, artifact_desc)
                    if verdict == ErrorVerdict.DLQ:
                        results[idx] = exc
                        if dlq is not None:
                            dlq.add(artifact_desc, exc, layer_name=layer_name)
                    else:
                        results[idx] = exc
                        if first_fatal is None:
                            first_fatal = exc

    # Flatten successful results in order
    all_artifacts: list[Artifact] = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        all_artifacts.extend(r)

    if first_fatal is not None:
        raise first_fatal

    return all_artifacts


def _materialize_search_surface(
    surface: SearchSurface,
    layer_artifacts: dict[str, list[Artifact]],
    work_dir: Path,
    logger: SynixLogger | None = None,
    triggered_by: str | None = None,
) -> None:
    """Materialize a build-time search surface to .synix/work/surfaces/."""
    source_layer_names = [s.name for s in surface.sources]

    all_artifacts: list[Artifact] = []
    sources_config: list[dict] = []

    for layer_name in source_layer_names:
        artifacts = layer_artifacts.get(layer_name, [])
        all_artifacts.extend(artifacts)
        for art in artifacts:
            level = art.metadata.get("layer_level", 0)
            sources_config.append({"layer": layer_name, "level": level})
            break

    if logger:
        logger.projection_start(surface.name, "search_surface", triggered_by=triggered_by)

    db_path = _surface_db_path(surface, work_dir)
    try:
        projection = get_projection("search_index", work_dir, db_path)
    except ValueError as exc:
        logger.warning("Surface %r unavailable: %s", surface.name, exc)
        return
    config = dict(surface.config)
    config["embedding_config"] = surface.embedding_config
    config["sources"] = sources_config
    if logger:
        config["_synix_logger"] = logger
    projection.materialize(all_artifacts, config)
    projection.close()

    if logger:
        logger.projection_finish(surface.name, triggered_by=triggered_by)
