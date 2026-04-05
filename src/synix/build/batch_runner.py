"""Batch build runner — walk DAG using OpenAI Batch API for eligible layers."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

from synix.build.batch_client import (
    BatchCollecting,
    BatchInProgress,
    BatchLLMClient,
    BatchRequestFailed,
)
from synix.build.batch_state import BatchState, BuildInstance
from synix.build.dag import compute_levels, needs_rebuild, resolve_build_order
from synix.build.fingerprint import compute_build_fingerprint
from synix.build.refs import synix_dir_for_build_dir
from synix.build.runner import (
    _build_source_config,
    _build_transform_config,
    _build_transform_context,
    _gather_inputs,
    _invoke_transform_execute,
    _invoke_transform_split,
    _layer_fully_cached,
    _record_snapshot_artifact,
    _record_snapshot_projections,
    _snapshot_parent_labels,
)
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.build.snapshots import commit_build_snapshot, start_build_transaction
from synix.core.config import LLMConfig
from synix.core.models import Artifact, Pipeline, Source, Transform

logger = logging.getLogger(__name__)


@dataclass
class BatchRunResult:
    """Summary of a batch build run."""

    status: str  # "submitted", "polling", "completed", "completed_with_errors", "failed"
    build_id: str
    layers_completed: list[str] = field(default_factory=list)
    layers_pending: list[str] = field(default_factory=list)
    batches_submitted: list[str] = field(default_factory=list)
    total_time: float = 0.0
    errors: dict[str, dict] = field(default_factory=dict)


def batch_run(
    pipeline: Pipeline,
    build_id: str,
    *,
    source_dir: str | None = None,
    poll: bool = False,
    poll_interval: int = 60,
    allow_pipeline_mismatch: bool = False,
    reset_state: bool = False,
) -> BatchRunResult:
    """Execute a pipeline using OpenAI Batch API for eligible layers.

    Args:
        pipeline: Pipeline definition.
        build_id: Unique identifier for this build instance.
        source_dir: Override for pipeline.source_dir.
        poll: Stay alive and poll until batch completes.
        poll_interval: Seconds between polls.
        allow_pipeline_mismatch: Resume despite pipeline fingerprint changes.
        reset_state: Clear corrupted state and restart current layer.

    Returns:
        BatchRunResult with build status and details.
    """
    start_time = time.time()
    src_dir = source_dir or pipeline.source_dir
    build_dir = Path(pipeline.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-validation ---
    cassette_mode = os.environ.get("SYNIX_CASSETTE_MODE", "off").lower()
    if cassette_mode == "replay" and not _has_cassette_responses():
        raise click.UsageError(
            "SYNIX_CASSETTE_MODE=replay is incompatible with batch-build "
            "unless batch_responses.json exists in SYNIX_CASSETTE_DIR."
        )

    compute_levels(pipeline.layers)
    build_order = resolve_build_order(pipeline)

    # Check for batchable layers
    has_batchable = False
    for layer in build_order:
        if isinstance(layer, Transform):
            mode = _resolve_batch_mode(layer, pipeline)
            if mode == "batch":
                has_batchable = True
                break
    if not has_batchable:
        raise click.UsageError(
            "No batchable layers found. All transform layers resolve to sync mode. "
            "Use 'synix build' for synchronous builds."
        )

    # --- Load or create build state ---
    try:
        batch_state = BatchState(build_dir, build_id)
    except RuntimeError:
        if reset_state:
            # State file was already quarantined by BatchState, just create fresh
            batch_state = BatchState.create_fresh(build_dir, build_id)
        else:
            raise

    manifest = batch_state.load_manifest()
    pipeline_hash = BatchState.compute_pipeline_hash(pipeline)

    if manifest is None:
        # New build
        manifest = BuildInstance(
            build_id=build_id,
            pipeline_hash=pipeline_hash,
            status="collecting",
        )
        batch_state.save_manifest(manifest)
    else:
        # Resuming — check pipeline fingerprint
        if manifest.pipeline_hash != pipeline_hash and not allow_pipeline_mismatch:
            raise click.UsageError(
                f"Pipeline fingerprint mismatch: build was created with "
                f"{manifest.pipeline_hash!r} but current pipeline is {pipeline_hash!r}. "
                f"Use --allow-pipeline-mismatch to resume anyway."
            )

    synix_dir = synix_dir_for_build_dir(build_dir)
    synix_dir.mkdir(parents=True, exist_ok=True)
    (synix_dir / "work").mkdir(parents=True, exist_ok=True)
    store = SnapshotArtifactCache(synix_dir)
    snapshot_txn = start_build_transaction(pipeline, build_dir, build_id)
    layer_artifacts: dict[str, list] = {}

    result = BatchRunResult(status="collecting", build_id=build_id)
    result.layers_completed = list(manifest.layers_completed)

    def _record_layer_artifacts(layer, inputs=None):
        """Record all artifacts for a completed layer into the snapshot transaction."""
        for art in layer_artifacts.get(layer.name, []):
            if isinstance(layer, Source):
                parent_labels = []
            else:
                parent_labels = _snapshot_parent_labels(art, inputs or [], store)
            _record_snapshot_artifact(
                snapshot_txn,
                art,
                layer_name=layer.name,
                layer_level=layer._level,
                parent_labels=parent_labels,
            )

    # --- DAG walk ---
    for layer in build_order:
        if layer.name in manifest.layers_completed:
            # Already done in previous run — load artifacts from store
            layer_artifacts[layer.name] = store.list_artifacts(layer.name)
            _record_layer_artifacts(layer)
            continue

        if isinstance(layer, Source):
            _run_source_layer(layer, pipeline, src_dir, store, layer_artifacts)
            _record_layer_artifacts(layer)
            manifest.layers_completed.append(layer.name)
            manifest.current_layer = None
            batch_state.save_manifest(manifest)

        elif isinstance(layer, Transform):
            manifest.current_layer = layer.name
            batch_state.save_manifest(manifest)

            mode = _resolve_batch_mode(layer, pipeline)
            inputs = _gather_inputs(layer, layer_artifacts, store)

            if mode == "sync":
                _run_sync_transform(layer, pipeline, src_dir, build_dir, inputs, store, layer_artifacts)
                _record_layer_artifacts(layer, inputs)
                manifest.layers_completed.append(layer.name)
                manifest.current_layer = None
                batch_state.save_manifest(manifest)
            else:
                # Batch mode
                batch_result = _run_batch_transform(
                    layer,
                    pipeline,
                    src_dir,
                    build_dir,
                    inputs,
                    store,
                    layer_artifacts,
                    batch_state,
                )

                if batch_result == "completed":
                    _record_layer_artifacts(layer, inputs)
                    manifest.layers_completed.append(layer.name)
                    manifest.current_layer = None
                    batch_state.save_manifest(manifest)
                elif batch_result == "submitted":
                    result.status = "submitted"
                    manifest.status = "submitted"
                    batch_state.save_manifest(manifest)

                    if poll:
                        # Poll loop
                        completed = _poll_and_resume(
                            layer,
                            pipeline,
                            src_dir,
                            build_dir,
                            inputs,
                            store,
                            layer_artifacts,
                            batch_state,
                            poll_interval,
                        )
                        if completed:
                            _record_layer_artifacts(layer, inputs)
                            manifest.layers_completed.append(layer.name)
                            manifest.current_layer = None
                            batch_state.save_manifest(manifest)
                        else:
                            result.layers_pending.append(layer.name)
                            result.total_time = time.time() - start_time
                            return result
                    else:
                        result.layers_pending.append(layer.name)
                        result.total_time = time.time() - start_time
                        return result
                elif batch_result == "in_progress":
                    result.status = "polling"
                    if poll:
                        completed = _poll_and_resume(
                            layer,
                            pipeline,
                            src_dir,
                            build_dir,
                            inputs,
                            store,
                            layer_artifacts,
                            batch_state,
                            poll_interval,
                        )
                        if completed:
                            _record_layer_artifacts(layer, inputs)
                            manifest.layers_completed.append(layer.name)
                            manifest.current_layer = None
                            batch_state.save_manifest(manifest)
                        else:
                            result.layers_pending.append(layer.name)
                            result.total_time = time.time() - start_time
                            return result
                    else:
                        result.layers_pending.append(layer.name)
                        result.total_time = time.time() - start_time
                        return result

    # All layers done — commit snapshot
    errors = batch_state.get_errors()
    if errors:
        result.status = "completed_with_errors"
        result.errors = errors
        manifest.status = "completed_with_errors"
        manifest.failed_requests = len(errors)
    else:
        result.status = "completed"
        manifest.status = "completed"

    _record_snapshot_projections(snapshot_txn, pipeline, layer_artifacts)
    commit_build_snapshot(snapshot_txn)

    result.layers_completed = list(manifest.layers_completed)
    result.batches_submitted = list(batch_state.get_batches().keys())
    result.total_time = time.time() - start_time
    batch_state.save_manifest(manifest)
    return result


def _resolve_batch_mode(layer: Transform, pipeline: Pipeline) -> str:
    """Determine whether a layer should run in batch or sync mode.

    Batch mode is only supported for ``provider="openai"`` with the default
    base URL (platform.openai.com).  Custom base URLs, DeepSeek, and
    openai-compatible providers are NOT supported — the OpenAI Batch API is
    a platform-specific feature.
    """
    if layer.batch is False:
        return "sync"

    # Resolve LLM config for this layer
    base_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
    if layer.config and layer.config.get("llm_config"):
        base_config.update(layer.config["llm_config"])
    llm_config = LLMConfig.from_dict(base_config)

    is_openai_native = llm_config.provider == "openai" and not llm_config.base_url

    if layer.batch is True:
        if llm_config.provider != "openai":
            raise ValueError(
                f"Layer {layer.name!r} has batch=True but uses provider "
                f"{llm_config.provider!r}. Batch mode requires provider='openai'."
            )
        if llm_config.base_url:
            raise ValueError(
                f"Layer {layer.name!r} has batch=True but sets a custom base_url "
                f"({llm_config.base_url!r}). The OpenAI Batch API only works with "
                f"the default OpenAI platform URL."
            )
        # Validate API key (skip when cassette responses exist — no real API needed)
        if not _has_cassette_responses():
            api_key = llm_config.resolve_api_key()
            if not api_key:
                raise ValueError(f"Layer {layer.name!r} requires OPENAI_API_KEY for batch mode.")
        return "batch"

    # batch=None (auto) — batch only if native OpenAI provider (no base_url override)
    # and the transform actually benefits from batching (produces multiple outputs)
    if is_openai_native:
        # N:1 transforms (estimate_output_count always 1) don't benefit from batching
        if layer.estimate_output_count(3) <= 1:
            return "sync"
        if not _has_cassette_responses():
            api_key = llm_config.resolve_api_key()
            if not api_key:
                raise ValueError(
                    f"Layer {layer.name!r} resolves to batch mode (OpenAI provider) but OPENAI_API_KEY is not set."
                )
        return "batch"

    return "sync"


def _run_source_layer(
    layer: Source,
    pipeline: Pipeline,
    src_dir: str,
    store: SnapshotArtifactCache,
    layer_artifacts: dict,
) -> None:
    """Run a source layer synchronously."""
    source_config = _build_source_config(pipeline, layer, src_dir)
    source_config["_layer_name"] = layer.name

    artifacts = layer.load(source_config)

    for artifact in artifacts:
        artifact.metadata["layer_name"] = layer.name
        artifact.metadata["layer_level"] = layer._level

    layer_artifacts[layer.name] = artifacts


def _save_or_cache_artifact(
    artifact,
    layer: Transform,
    inputs: list,
    store: SnapshotArtifactCache,
    transform_fp,
) -> Artifact:
    """Check cache and return the final artifact (cached or newly built)."""

    build_fp = compute_build_fingerprint(transform_fp, artifact.input_ids)
    rebuild, _reasons = needs_rebuild(
        artifact.label,
        artifact.input_ids,
        store,
        current_build_fingerprint=build_fp,
    )
    if rebuild:
        artifact.metadata["layer_name"] = layer.name
        artifact.metadata["layer_level"] = layer._level
        artifact.metadata["build_fingerprint"] = build_fp.to_dict()
        artifact.metadata["transform_fingerprint"] = transform_fp.to_dict()
        return artifact
    else:
        cached = store.load_artifact(artifact.label)
        if cached is not None:
            cached.metadata["layer_name"] = layer.name
            cached.metadata["layer_level"] = layer._level
            return cached
        return artifact


def _run_sync_transform(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: SnapshotArtifactCache,
    layer_artifacts: dict,
) -> None:
    """Run a transform layer synchronously (same as regular build)."""
    transform_config = _build_transform_config(pipeline, layer, src_dir, build_dir)
    transform_ctx = _build_transform_context(pipeline, layer, src_dir, build_dir, transform_config)
    transform_fp = layer.compute_fingerprint(transform_config)

    layer_built = []

    if _layer_fully_cached(layer, inputs, store, transform_fp):
        existing = store.list_artifacts(layer.name)
        for art in existing:
            art.metadata["layer_name"] = layer.name
            art.metadata["layer_level"] = layer._level
            layer_built.append(art)
    else:
        transform_ctx = transform_ctx.with_updates({"_layer_name": layer.name})
        units = _invoke_transform_split(layer, inputs, transform_ctx)
        for unit_inputs, config_extras in units:
            unit_ctx = transform_ctx.with_updates(config_extras)
            new_artifacts = _invoke_transform_execute(layer, unit_inputs, unit_ctx)
            for artifact in new_artifacts:
                layer_built.append(_save_or_cache_artifact(artifact, layer, inputs, store, transform_fp))

    layer_artifacts[layer.name] = layer_built


def _run_batch_transform(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: SnapshotArtifactCache,
    layer_artifacts: dict,
    batch_state: BatchState,
) -> str:
    """Run a transform layer in batch mode.

    Returns:
        "completed" if all results available and artifacts created.
        "submitted" if a new batch was submitted.
        "in_progress" if a batch is already in progress.
    """
    transform_config = _build_transform_config(pipeline, layer, src_dir, build_dir)
    transform_ctx = _build_transform_context(pipeline, layer, src_dir, build_dir, transform_config)
    transform_fp = layer.compute_fingerprint(transform_config)

    if _layer_fully_cached(layer, inputs, store, transform_fp):
        existing = store.list_artifacts(layer.name)
        for art in existing:
            art.metadata["layer_name"] = layer.name
            art.metadata["layer_level"] = layer._level
        layer_artifacts[layer.name] = existing
        return "completed"

    # Resolve LLM config for BatchLLMClient
    base_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
    if layer.config and layer.config.get("llm_config"):
        base_config.update(layer.config["llm_config"])
    llm_config = LLMConfig.from_dict(base_config)

    # Load cassette responses if available
    cassette_responses = _load_cassette_responses(build_dir)

    batch_client = BatchLLMClient(llm_config, batch_state, layer.name, cassette_responses)

    transform_ctx = transform_ctx.with_updates({"_layer_name": layer.name, "_shared_llm_client": batch_client})

    units = _invoke_transform_split(layer, inputs, transform_ctx)

    layer_built = []
    collecting = False
    in_progress = False

    failed_units: list[str] = []
    for unit_inputs, config_extras in units:
        unit_ctx = transform_ctx.with_updates(config_extras)
        try:
            new_artifacts = _invoke_transform_execute(layer, unit_inputs, unit_ctx)
            # Result was available — save artifacts
            for artifact in new_artifacts:
                layer_built.append(_save_or_cache_artifact(artifact, layer, inputs, store, transform_fp))
        except BatchCollecting:
            collecting = True
        except BatchInProgress:
            in_progress = True
            break
        except BatchRequestFailed as exc:
            # Record per-unit failure and continue with remaining units
            unit_desc = ", ".join(a.label for a in unit_inputs) if unit_inputs else "unknown"
            logger.warning("Batch request failed for layer %r unit [%s]: %s", layer.name, unit_desc, exc)
            failed_units.append(unit_desc)

    if in_progress:
        layer_artifacts[layer.name] = layer_built
        return "in_progress"

    if collecting:
        # Submit the batch
        pending = batch_state.get_pending(layer.name)
        if pending:
            batch_id = batch_client.submit_batch(layer.name)
            logger.info("Submitted batch %s for layer %s", batch_id, layer.name)
        layer_artifacts[layer.name] = layer_built
        return "submitted"

    # Report per-unit failures (non-fatal — other units still processed)
    if failed_units:
        for unit_desc in failed_units:
            batch_state.store_error(
                f"{layer.name}:{unit_desc}",
                "batch_request_failed",
                f"Batch request failed for unit [{unit_desc}] in layer {layer.name!r}",
            )
        batch_state.save()

    # All results were available (some may have failed)
    layer_artifacts[layer.name] = layer_built
    return "completed"


def _poll_and_resume(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: SnapshotArtifactCache,
    layer_artifacts: dict,
    batch_state: BatchState,
    poll_interval: int,
) -> bool:
    """Poll for batch completion then re-run the layer.

    Returns True if layer completed, False if still pending.
    """
    base_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
    if layer.config and layer.config.get("llm_config"):
        base_config.update(layer.config["llm_config"])
    llm_config = LLMConfig.from_dict(base_config)

    cassette_responses = _load_cassette_responses(build_dir)
    batch_client = BatchLLMClient(llm_config, batch_state, layer.name, cassette_responses)

    # Find active batches for this layer
    batches = batch_state.get_batches(layer.name)
    active_batch_ids = [
        bid for bid, b in batches.items() if b["status"] not in ("completed", "failed", "expired", "cancelled")
    ]

    max_duration = 86400  # 24 hours
    poll_start = time.time()
    while (time.time() - poll_start) < max_duration:
        all_done = True
        for batch_id in active_batch_ids:
            try:
                done = batch_client.check_and_download(batch_id)
                if not done:
                    all_done = False
            except Exception as exc:
                raise RuntimeError(f"Error checking batch {batch_id}: {exc}") from exc

        if all_done:
            # Re-run the layer to produce artifacts from cached results
            result = _run_batch_transform(
                layer,
                pipeline,
                src_dir,
                build_dir,
                inputs,
                store,
                layer_artifacts,
                batch_state,
            )
            return result == "completed"

        time.sleep(poll_interval)

    return False


def _load_cassette_responses(build_dir: Path) -> dict | None:
    """Load pre-seeded batch responses from cassettes/batch_responses.json."""
    cassette_dir = os.environ.get("SYNIX_CASSETTE_DIR")
    if not cassette_dir:
        return None
    responses_path = Path(cassette_dir) / "batch_responses.json"
    if not responses_path.exists():
        return None
    # Fail hard on parse errors — corrupted cassette file must not be silently ignored
    return json.loads(responses_path.read_text())


def _has_cassette_responses() -> bool:
    """Check if batch_responses.json exists in cassette dir."""
    cassette_dir = os.environ.get("SYNIX_CASSETTE_DIR")
    if not cassette_dir:
        return False
    return (Path(cassette_dir) / "batch_responses.json").exists()


def plan_batch(pipeline: Pipeline) -> list[dict]:
    """Dry-run: show which layers would batch vs sync, estimated request counts.

    Returns a list of layer info dicts.
    """
    compute_levels(pipeline.layers)
    build_order = resolve_build_order(pipeline)

    layer_cardinality: dict[str, int] = {}
    layers_info = []
    for layer in build_order:
        info: dict = {
            "name": layer.name,
            "level": layer._level,
            "type": type(layer).__name__,
        }

        if isinstance(layer, Source):
            info["mode"] = "source"
            try:
                source_config = {"source_dir": pipeline.source_dir}
                artifacts = layer.load(source_config)
                layer_cardinality[layer.name] = len(artifacts)
            except Exception as exc:
                logger.warning(
                    "Source '%s' failed to load during batch plan: %s",
                    layer.name,
                    exc,
                )
                layer_cardinality[layer.name] = 1
        elif isinstance(layer, Transform):
            try:
                mode = _resolve_batch_mode(layer, pipeline)
            except ValueError as exc:
                # Missing API key is OK in plan mode — show as sync.
                # Provider/base_url misconfiguration should still fail loud.
                if "API_KEY" in str(exc):
                    mode = "sync"
                else:
                    raise
            except click.UsageError:
                mode = "sync"
            info["mode"] = mode
            info["batch_param"] = layer.batch

            # Estimate work units using DAG-aware cardinality tracking
            dep_counts = sum(layer_cardinality.get(dep.name, 1) for dep in layer.depends_on)
            if dep_counts == 0:
                dep_counts = 1
            info["estimated_requests"] = layer.estimate_output_count(dep_counts)
            layer_cardinality[layer.name] = layer.estimate_output_count(dep_counts)
        else:
            info["mode"] = "unknown"

        layers_info.append(info)

    return layers_info
