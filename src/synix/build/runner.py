"""Pipeline runner — walk DAG, run transforms, cache artifacts."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from synix.core.models import Artifact, Layer, Pipeline, Projection
from synix.core.logging import SynixLogger, Verbosity
from synix.build.provenance import ProvenanceTracker
from synix.build.artifacts import ArtifactStore
from synix.build.dag import needs_rebuild, resolve_build_order
from synix.build.projections import get_projection, FlatFileProjection
from synix.build.transforms import get_transform

# Import transform modules to trigger @register_transform decorators
import synix.build.parse_transform  # noqa: F401
import synix.build.llm_transforms  # noqa: F401
import synix.build.merge_transform  # noqa: F401


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
class RunResult:
    """Summary of a pipeline run."""

    built: int = 0
    cached: int = 0
    skipped: int = 0
    total_time: float = 0.0
    layer_stats: list[LayerStats] = field(default_factory=list)
    run_log: dict = field(default_factory=dict)


def run(pipeline: Pipeline, source_dir: str | None = None, verbosity: int = 0,
        concurrency: int = 1) -> RunResult:
    """Execute the full pipeline — walk DAG, run transforms, materialize projections.

    Args:
        pipeline: The pipeline definition.
        source_dir: Override for pipeline.source_dir.
        verbosity: Verbosity level (0=default, 1=verbose, 2=debug).
        concurrency: Number of concurrent LLM requests (1 = sequential).

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
    )

    # Resolve build order
    build_order = resolve_build_order(pipeline)
    logger.run_start(pipeline.name, len(build_order))

    # Build a lookup from layer name to layer for levels
    layer_map = {layer.name: layer for layer in pipeline.layers}

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

            # Determine whether to run transforms concurrently.
            # For "by_conversation" grouping (1:1 input->output, e.g. episode summaries),
            # we can split inputs and process them in parallel.
            # For other groupings (by_month, by_topic, single), the transform needs
            # all inputs together for internal grouping, so we run sequentially.
            use_concurrent = (
                concurrency > 1
                and layer.level > 0
                and layer.grouping == "by_conversation"
                and len(inputs) > 1
            )

            if use_concurrent:
                new_artifacts = _execute_transform_concurrent(
                    transform, inputs, transform_config, concurrency
                )
            else:
                new_artifacts = transform.execute(inputs, transform_config)

            # Check rebuild for each artifact
            for artifact in new_artifacts:
                if needs_rebuild(
                    artifact.artifact_id, artifact.input_hashes, prompt_id, store,
                    current_model_config=model_config,
                    current_transform_cache_key=transform_cache_key,
                ):
                    # Set layer metadata
                    artifact.metadata["layer_name"] = layer.name
                    artifact.metadata["layer_level"] = layer.level
                    if transform_cache_key:
                        artifact.metadata["transform_cache_key"] = transform_cache_key

                    # Save and record provenance
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
                    # Load cached version
                    cached = store.load_artifact(artifact.artifact_id)
                    if cached is not None:
                        cached.metadata["layer_name"] = layer.name
                        cached.metadata["layer_level"] = layer.level
                        layer_built.append(cached)
                    else:
                        layer_built.append(artifact)
                    stats.cached += 1
                    logger.artifact_cached(layer.name, artifact.artifact_id)

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
            pipeline, layer.name, layer_artifacts, store, build_dir
        )

    # Materialize all final projections
    _materialize_all_projections(pipeline, layer_artifacts, store, build_dir)

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
    inputs: list[Artifact],
    config: dict,
    concurrency: int,
) -> list[Artifact]:
    """Execute a 1:1 transform concurrently across inputs.

    Splits inputs into individual items and runs transform.execute([item], config)
    in parallel using a thread pool. Results are collected and flattened in the
    original input order.

    Args:
        transform: The transform instance to execute.
        inputs: List of input artifacts (each processed independently).
        config: Transform configuration dict.
        concurrency: Maximum number of concurrent workers.

    Returns:
        List of output artifacts in the same order as inputs.
    """
    results: list[list[Artifact] | Exception] = [None] * len(inputs)  # type: ignore[list-item]

    def _run_one(index: int, single_input: Artifact) -> tuple[int, list[Artifact]]:
        """Execute the transform for a single input, returning (index, artifacts)."""
        return index, transform.execute([single_input], config)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, i, inp): i
            for i, inp in enumerate(inputs)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, artifacts = future.result()
                results[idx] = artifacts
            except Exception as exc:
                results[idx] = exc

    # Flatten results in order, re-raising the first error encountered
    all_artifacts: list[Artifact] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise RuntimeError(
                f"Concurrent transform produced no result for input {i}"
            )
        all_artifacts.extend(result)

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
) -> None:
    """Materialize projections that source from this layer."""
    for proj in pipeline.projections:
        source_layers = [s["layer"] for s in proj.sources]
        if layer_name not in source_layers:
            continue

        if not all(ln in layer_artifacts for ln in source_layers):
            continue

        _materialize_projection(proj, layer_artifacts, build_dir)


def _materialize_all_projections(
    pipeline: Pipeline,
    layer_artifacts: dict[str, list[Artifact]],
    store: ArtifactStore,
    build_dir: Path,
) -> None:
    """Materialize all projections after the full build."""
    for proj in pipeline.projections:
        _materialize_projection(proj, layer_artifacts, build_dir)


def _materialize_projection(
    proj: Projection,
    layer_artifacts: dict[str, list[Artifact]],
    build_dir: Path,
) -> None:
    """Materialize a single projection."""
    all_artifacts: list[Artifact] = []
    sources_config: list[dict] = []

    for source in proj.sources:
        layer_name = source["layer"]
        artifacts = layer_artifacts.get(layer_name, [])
        all_artifacts.extend(artifacts)
        for art in artifacts:
            level = art.metadata.get("layer_level", 0)
            sources_config.append({"layer": layer_name, "level": level})
            break

    if proj.projection_type == "search_index":
        # Use projection registry to avoid direct search import
        try:
            projection = get_projection("search_index", build_dir)
        except ValueError:
            # Search module not loaded — skip silently
            return
        config = {"sources": sources_config}
        projection.materialize(all_artifacts, config)
        projection.close()
    elif proj.projection_type == "flat_file":
        projection = FlatFileProjection()
        config = dict(proj.config)
        if "output_path" not in config:
            config["output_path"] = str(build_dir / "context.md")
        projection.materialize(all_artifacts, config)
