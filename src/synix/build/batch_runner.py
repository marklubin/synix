"""Batch build runner — walk DAG using OpenAI Batch API for eligible layers."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

from synix.build.artifacts import ArtifactStore
from synix.build.batch_client import (
    BatchCollecting,
    BatchInProgress,
    BatchLLMClient,
    BatchRequestFailed,
)
from synix.build.batch_state import BatchState, BuildInstance
from synix.build.dag import compute_levels, needs_rebuild, resolve_build_order
from synix.build.fingerprint import compute_build_fingerprint
from synix.build.provenance import ProvenanceTracker
from synix.build.runner import (
    _build_source_config,
    _build_transform_config,
    _gather_inputs,
    _get_parent_labels,
    _layer_fully_cached,
)
from synix.core.config import LLMConfig
from synix.core.models import Pipeline, Source, Transform

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
            # Clear corrupted state
            instance_dir = build_dir / "builds" / build_id
            state_path = instance_dir / "batch_state.json"
            # State file was already quarantined by BatchState, just create fresh
            batch_state = BatchState.__new__(BatchState)
            batch_state.build_dir = build_dir
            batch_state.build_id = build_id
            batch_state._instance_dir = instance_dir
            batch_state._manifest_path = instance_dir / "manifest.json"
            batch_state._state_path = state_path
            batch_state._pending = {}
            batch_state._batch_map = {}
            batch_state._batches = {}
            batch_state._results = {}
            batch_state._errors = {}
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

    store = ArtifactStore(build_dir)
    provenance = ProvenanceTracker(build_dir)
    layer_artifacts: dict[str, list] = {}

    result = BatchRunResult(status="collecting", build_id=build_id)
    result.layers_completed = list(manifest.layers_completed)

    # --- DAG walk ---
    for layer in build_order:
        if layer.name in manifest.layers_completed:
            # Already done in previous run — load artifacts from store
            layer_artifacts[layer.name] = store.list_artifacts(layer.name)
            continue

        if isinstance(layer, Source):
            _run_source_layer(layer, pipeline, src_dir, store, provenance, layer_artifacts)
            manifest.layers_completed.append(layer.name)
            manifest.current_layer = None
            batch_state.save_manifest(manifest)

        elif isinstance(layer, Transform):
            manifest.current_layer = layer.name
            batch_state.save_manifest(manifest)

            mode = _resolve_batch_mode(layer, pipeline)
            inputs = _gather_inputs(layer, layer_artifacts, store)

            if mode == "sync":
                _run_sync_transform(layer, pipeline, src_dir, build_dir, inputs, store, provenance, layer_artifacts)
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
                    provenance,
                    layer_artifacts,
                    batch_state,
                )

                if batch_result == "completed":
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
                            provenance,
                            layer_artifacts,
                            batch_state,
                            poll_interval,
                        )
                        if completed:
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
                            provenance,
                            layer_artifacts,
                            batch_state,
                            poll_interval,
                        )
                        if completed:
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

    # All layers done
    errors = batch_state.get_errors()
    if errors:
        result.status = "completed_with_errors"
        result.errors = errors
        manifest.status = "completed_with_errors"
        manifest.failed_requests = len(errors)
    else:
        result.status = "completed"
        manifest.status = "completed"

    result.layers_completed = list(manifest.layers_completed)
    result.batches_submitted = list(batch_state.get_batches().keys())
    result.total_time = time.time() - start_time
    batch_state.save_manifest(manifest)
    return result


def _resolve_batch_mode(layer: Transform, pipeline: Pipeline) -> str:
    """Determine whether a layer should run in batch or sync mode."""
    if layer.batch is False:
        return "sync"

    # Resolve LLM config for this layer
    base_config = dict(pipeline.llm_config) if pipeline.llm_config else {}
    if layer.config and layer.config.get("llm_config"):
        base_config.update(layer.config["llm_config"])
    llm_config = LLMConfig.from_dict(base_config)

    is_openai = llm_config.provider in ("openai", "deepseek", "openai-compatible")

    if layer.batch is True:
        if not is_openai:
            raise ValueError(
                f"Layer {layer.name!r} has batch=True but uses provider "
                f"{llm_config.provider!r}. Batch mode requires an OpenAI-compatible provider."
            )
        # Validate API key (skip when cassette responses exist — no real API needed)
        if not _has_cassette_responses():
            api_key = llm_config.resolve_api_key()
            if not api_key:
                raise ValueError(f"Layer {layer.name!r} requires OPENAI_API_KEY for batch mode.")
        return "batch"

    # batch=None (auto) — batch if OpenAI provider
    if is_openai:
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
    store: ArtifactStore,
    provenance: ProvenanceTracker,
    layer_artifacts: dict,
) -> None:
    """Run a source layer synchronously."""
    source_config = _build_source_config(pipeline, layer, src_dir)
    source_config["_layer_name"] = layer.name

    try:
        artifacts = layer.load(source_config)
    except Exception:
        logger.warning("Source %s failed to load", layer.name, exc_info=True)
        artifacts = []

    for artifact in artifacts:
        artifact.metadata["layer_name"] = layer.name
        artifact.metadata["layer_level"] = layer._level
        store.save_artifact(artifact, layer.name, layer._level)
        provenance.record(artifact.label, parent_labels=[], prompt_id=None, model_config=None)

    layer_artifacts[layer.name] = artifacts


def _run_sync_transform(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
    layer_artifacts: dict,
) -> None:
    """Run a transform layer synchronously (same as regular build)."""
    transform_config = _build_transform_config(pipeline, layer, src_dir, build_dir)
    transform_fp = layer.compute_fingerprint(transform_config)

    layer_built = []

    if _layer_fully_cached(layer, inputs, store, transform_fp):
        existing = store.list_artifacts(layer.name)
        for art in existing:
            art.metadata["layer_name"] = layer.name
            art.metadata["layer_level"] = layer._level
            layer_built.append(art)
    else:
        transform_config["_layer_name"] = layer.name
        units = layer.split(inputs, transform_config)
        for unit_inputs, config_extras in units:
            merged_config = {**transform_config, **config_extras}
            new_artifacts = layer.execute(unit_inputs, merged_config)
            for artifact in new_artifacts:
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
                    store.save_artifact(artifact, layer.name, layer._level)
                    parent_labels = _get_parent_labels(artifact, inputs)
                    provenance.record(
                        artifact.label,
                        parent_labels=parent_labels,
                        prompt_id=artifact.prompt_id,
                        model_config=artifact.model_config,
                    )
                    layer_built.append(artifact)
                else:
                    cached = store.load_artifact(artifact.label)
                    if cached is not None:
                        cached.metadata["layer_name"] = layer.name
                        cached.metadata["layer_level"] = layer._level
                        layer_built.append(cached)
                    else:
                        layer_built.append(artifact)

    layer_artifacts[layer.name] = layer_built


def _run_batch_transform(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
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

    transform_config["_layer_name"] = layer.name
    transform_config["_shared_llm_client"] = batch_client

    units = layer.split(inputs, transform_config)

    layer_built = []
    collecting = False
    in_progress = False

    for unit_inputs, config_extras in units:
        merged_config = {**transform_config, **config_extras}
        try:
            new_artifacts = layer.execute(unit_inputs, merged_config)
            # Result was available — save artifacts
            for artifact in new_artifacts:
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
                    store.save_artifact(artifact, layer.name, layer._level)
                    parent_labels = _get_parent_labels(artifact, inputs)
                    provenance.record(
                        artifact.label,
                        parent_labels=parent_labels,
                        prompt_id=artifact.prompt_id,
                        model_config=artifact.model_config,
                    )
                    layer_built.append(artifact)
                else:
                    cached = store.load_artifact(artifact.label)
                    if cached is not None:
                        layer_built.append(cached)
                    else:
                        layer_built.append(artifact)
        except BatchCollecting:
            collecting = True
        except BatchInProgress:
            in_progress = True
            break
        except BatchRequestFailed as exc:
            logger.warning("Request failed for layer %s: %s", layer.name, exc)

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

    # All results were available
    layer_artifacts[layer.name] = layer_built
    return "completed"


def _poll_and_resume(
    layer: Transform,
    pipeline: Pipeline,
    src_dir: str,
    build_dir: Path,
    inputs: list,
    store: ArtifactStore,
    provenance: ProvenanceTracker,
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

    max_polls = 1440  # 24h at 60s intervals
    for _ in range(max_polls):
        all_done = True
        for batch_id in active_batch_ids:
            try:
                done = batch_client.check_and_download(batch_id)
                if not done:
                    all_done = False
            except Exception as exc:
                logger.warning("Error checking batch %s: %s", batch_id, exc)
                all_done = False

        if all_done:
            # Re-run the layer to produce artifacts from cached results
            result = _run_batch_transform(
                layer,
                pipeline,
                src_dir,
                build_dir,
                inputs,
                store,
                provenance,
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
    try:
        return json.loads(responses_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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

    layers_info = []
    for layer in build_order:
        info: dict = {
            "name": layer.name,
            "level": layer._level,
            "type": type(layer).__name__,
        }

        if isinstance(layer, Source):
            info["mode"] = "source"
        elif isinstance(layer, Transform):
            try:
                mode = _resolve_batch_mode(layer, pipeline)
            except (ValueError, click.UsageError):
                mode = "sync"
            info["mode"] = mode
            info["batch_param"] = layer.batch

            # Estimate work units
            dep_counts = sum(
                dep_layer.estimate_output_count(1) for dep_layer in layer.depends_on if isinstance(dep_layer, Transform)
            )
            if dep_counts == 0:
                dep_counts = len(layer.depends_on) or 1
            info["estimated_requests"] = layer.estimate_output_count(dep_counts)
        else:
            info["mode"] = "unknown"

        layers_info.append(info)

    return layers_info
