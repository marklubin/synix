"""Pipeline class - the core API surface for Synix."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from synix.config import Settings
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient
    from synix.services.search import SearchHit
    from synix.sources.base import Source
    from synix.steps.base import Step
    from synix.surfaces.base import Surface

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a pipeline run."""

    run_id: str
    stats: dict[str, int]
    status: str = "completed"
    error: str | None = None


@dataclass
class PlanResult:
    """Result from pipeline.plan() - shows what would execute."""

    steps: list[dict[str, Any]]
    total_inputs: int
    would_process: int
    would_skip: int


@dataclass
class ArtifactStep:
    """Internal representation of an artifact publishing step."""

    name: str
    from_: list[str]  # Source step names
    surface: "Surface"


@dataclass
class Pipeline:
    """Declarative pipeline for processing AI conversation exports.

    Usage:
        pipeline = Pipeline("personal-memory", agent="mark")
        pipeline.source("claude", file="~/exports/claude.json", format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=summarize)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=reflect)
        pipeline.fold("narrative", from_="monthly", prompt=evolve)
        pipeline.artifact("report", from_="narrative", surface="file://output/report.md")
        pipeline.run()
    """

    name: str
    agent: str
    settings: "Settings | None" = None
    llm: "LLMClient | None" = None

    # Internal state
    _sources: dict[str, "Source"] = field(default_factory=dict)
    _steps: dict[str, "Step"] = field(default_factory=dict)
    _artifacts: dict[str, ArtifactStep] = field(default_factory=dict)
    _execution_order: list[str] = field(default_factory=list)
    _initialized: bool = field(default=False)

    def __post_init__(self) -> None:
        """Initialize settings if not provided."""
        if self.settings is None:
            from synix.config import get_settings

            self.settings = get_settings()

    def _ensure_initialized(self) -> None:
        """Ensure databases are initialized."""
        if not self._initialized:
            from synix.db.engine import init_databases

            init_databases(self.settings)
            self._initialized = True

    def source(
        self,
        name: str,
        *,
        file: str,
        format: str,
    ) -> None:
        """Register a source step.

        Args:
            name: Step name for this source.
            file: Path to the source file.
            format: Format identifier ('claude-export', 'chatgpt-export').
        """
        from synix.sources.base import expand_path
        from synix.sources.chatgpt import ChatGPTExportSource
        from synix.sources.claude import ClaudeExportSource

        file_path = expand_path(file)

        if format == "claude-export":
            source = ClaudeExportSource(name=name, file_path=file_path)
        elif format == "chatgpt-export":
            source = ChatGPTExportSource(name=name, file_path=file_path)
        else:
            msg = f"Unknown source format: {format}"
            raise ValueError(msg)

        self._sources[name] = source
        self._invalidate_order()

    def transform(
        self,
        name: str,
        *,
        from_: str,
        prompt: Callable[["Record"], str],
        model: str = "deepseek-chat",
    ) -> None:
        """Register a transform step (1:1 processing).

        Args:
            name: Step name.
            from_: Name of upstream step.
            prompt: Function that takes a Record and returns a prompt string.
            model: LLM model to use.
        """
        from synix.steps.transform import TransformStep

        step = TransformStep(
            name=name,
            from_=from_,
            prompt=prompt,
            model=model,
        )
        self._steps[name] = step
        self._invalidate_order()

    def aggregate(
        self,
        name: str,
        *,
        from_: str,
        period: str,
        prompt: Callable[[list["Record"], str], str],
        model: str = "deepseek-chat",
    ) -> None:
        """Register an aggregate step (N:1 processing).

        Args:
            name: Step name.
            from_: Name of upstream step.
            period: Grouping period ('month', 'week', 'day').
            prompt: Function that takes (records, period) and returns a prompt string.
            model: LLM model to use.
        """
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name=name,
            from_=from_,
            prompt=prompt,
            period=period,
            model=model,
        )
        self._steps[name] = step
        self._invalidate_order()

    def fold(
        self,
        name: str,
        *,
        from_: str,
        prompt: Callable[[str, "Record"], str],
        initial_state: str = "",
        model: str = "deepseek-chat",
    ) -> None:
        """Register a fold step (sequential processing with state accumulation).

        Processes records sequentially, carrying state forward. Each record
        transforms the accumulated state via LLM. Final output is the
        accumulated result.

        Args:
            name: Step name.
            from_: Name of upstream step.
            prompt: Function that takes (state, Record) and returns a prompt string.
            initial_state: Initial accumulator value.
            model: LLM model to use.
        """
        from synix.steps.fold import FoldStep

        step = FoldStep(
            name=name,
            from_=from_,
            prompt=prompt,
            initial_state=initial_state,
            model=model,
        )
        self._steps[name] = step
        self._invalidate_order()

    def merge(
        self,
        name: str,
        *,
        sources: list[str],
        prompt: Callable[[dict[str, list["Record"]]], str],
        model: str = "deepseek-chat",
    ) -> None:
        """Register a merge step (multi-source combination).

        Combines records from multiple upstream sources into a single output.

        Args:
            name: Step name.
            sources: List of upstream step names to merge.
            prompt: Function that takes dict[str, list[Record]] and returns a prompt string.
            model: LLM model to use.
        """
        from synix.steps.merge import MergeStep

        step = MergeStep(
            name=name,
            from_=None,
            sources=sources,
            prompt=prompt,
            model=model,
        )
        self._steps[name] = step
        self._invalidate_order()

    def artifact(
        self,
        name: str,
        *,
        from_: str | list[str],
        surface: str,
    ) -> None:
        """Register an artifact publishing step.

        Artifacts are published after all pipeline steps complete.

        Args:
            name: Artifact name.
            from_: Upstream step(s) to publish.
            surface: Surface URI (e.g., 'file://output/report.md').
        """
        from synix.surfaces.file import parse_file_surface

        # Normalize from_ to list
        source_steps = [from_] if isinstance(from_, str) else from_

        # Parse surface URI
        if surface.startswith("file://"):
            surface_obj = parse_file_surface(name, surface)
        else:
            msg = f"Unknown surface type: {surface}"
            raise ValueError(msg)

        self._artifacts[name] = ArtifactStep(
            name=name,
            from_=source_steps,
            surface=surface_obj,
        )

    def _invalidate_order(self) -> None:
        """Clear cached execution order."""
        self._execution_order = []

    def _resolve_order(self) -> list[str]:
        """Resolve step execution order using topological sort.

        Returns:
            List of step names in execution order.
        """
        from synix.steps.merge import MergeStep

        if self._execution_order:
            return self._execution_order

        # Build dependency graph
        deps: dict[str, set[str]] = {}

        # Sources have no dependencies
        for name in self._sources:
            deps[name] = set()

        # Steps depend on their from_ step (or sources for MergeStep)
        for name, step in self._steps.items():
            if isinstance(step, MergeStep):
                # MergeStep depends on all its sources
                deps[name] = set(step.sources)
            elif step.from_:
                deps[name] = {step.from_}
            else:
                deps[name] = set()

        # Topological sort (Kahn's algorithm)
        order: list[str] = []
        ready = [n for n, d in deps.items() if not d]

        while ready:
            node = ready.pop(0)
            order.append(node)

            # Remove this node from dependencies
            for name, dep_set in deps.items():
                if node in dep_set:
                    dep_set.remove(node)
                    if not dep_set and name not in order and name not in ready:
                        ready.append(name)

        # Check for cycles
        if len(order) != len(deps):
            remaining = set(deps.keys()) - set(order)
            msg = f"Cycle detected in step dependencies: {remaining}"
            raise ValueError(msg)

        self._execution_order = order
        return order

    def _get_llm(self) -> "LLMClient":
        """Get or create LLM client."""
        if self.llm is not None:
            return self.llm

        from synix.llm.client import LLMClient

        return LLMClient.from_settings()

    def run(
        self,
        step: str | None = None,
        *,
        full: bool = False,
    ) -> RunResult:
        """Execute the pipeline.

        Args:
            step: Optional specific step to run (and its dependencies).
            full: Force full reprocessing (ignore cache).

        Returns:
            RunResult with execution statistics.
        """
        from synix.db.engine import get_artifact_session, get_control_session
        from synix.services import pipelines, runs

        self._ensure_initialized()

        # Resolve execution order
        order = self._resolve_order()
        if step:
            order = self._filter_order_for_step(order, step)

        stats = {"input": 0, "output": 0, "skipped": 0, "errors": 0, "tokens": 0}

        # Save pipeline definition and create run record
        with get_control_session(self.settings) as control_session:
            # Save pipeline definition first (Run has FK to PipelineState)
            pipelines.save_pipeline(
                control_session,
                self.name,
                self.agent,
                {"sources": list(self._sources.keys()), "steps": list(self._steps.keys())},
            )
            control_session.flush()  # Ensure pipeline is persisted before run

            run = runs.create_run(control_session, self.name)
            run_id = str(run.id)

        # Execute steps
        llm = self._get_llm()

        try:
            with get_control_session(self.settings) as control_session:
                runs.start_run(control_session, run_id)

            with get_artifact_session(self.settings) as artifact_session:
                for step_name in order:
                    step_stats = self._execute_step(
                        step_name,
                        artifact_session,
                        run_id,
                        llm,
                        full=full,
                    )

                    stats["input"] += step_stats.get("input", 0)
                    stats["output"] += step_stats.get("output", 0)
                    stats["skipped"] += step_stats.get("skipped", 0)
                    stats["errors"] += step_stats.get("errors", 0)
                    stats["tokens"] += step_stats.get("tokens", 0)

                # Publish artifacts after all steps complete
                for _artifact_name, artifact_step in self._artifacts.items():
                    self._publish_artifact(artifact_step, artifact_session, run_id)

            with get_control_session(self.settings) as control_session:
                runs.complete_run(control_session, run_id, stats)

            return RunResult(run_id=run_id, stats=stats)

        except Exception as e:
            with get_control_session(self.settings) as control_session:
                runs.fail_run(control_session, run_id, str(e))
            return RunResult(run_id=run_id, stats=stats, status="failed", error=str(e))

    def _filter_order_for_step(self, order: list[str], target: str) -> list[str]:
        """Filter execution order to only include steps needed for target."""
        from synix.steps.merge import MergeStep

        if target not in order:
            msg = f"Step '{target}' not found"
            raise ValueError(msg)

        # Find all dependencies recursively
        needed: set[str] = set()

        def add_deps(name: str) -> None:
            if name in needed:
                return
            needed.add(name)

            # Check if it's a step with dependencies
            if name in self._steps:
                step = self._steps[name]
                if isinstance(step, MergeStep):
                    # MergeStep depends on all its sources
                    for source in step.sources:
                        add_deps(source)
                elif step.from_:
                    add_deps(step.from_)

        add_deps(target)

        # Return order filtered to only needed steps
        return [s for s in order if s in needed]

    def _publish_artifact(
        self,
        artifact_step: ArtifactStep,
        artifact_session: "Session",
        run_id: str,
    ) -> None:
        """Publish an artifact to its surface.

        Args:
            artifact_step: Artifact definition.
            artifact_session: Database session.
            run_id: Current run ID.
        """
        from synix.services import records

        # Gather records from all source steps
        all_records: list["Record"] = []
        for source_step in artifact_step.from_:
            step_records = records.get_records_by_step(artifact_session, source_step)
            all_records.extend(step_records)

        # Publish
        result = artifact_step.surface.publish(all_records, run_id)

        if result.success:
            logger.info(
                "Published artifact '%s' to %s (%d records)",
                artifact_step.name,
                result.location,
                result.count,
            )
        else:
            logger.error(
                "Failed to publish artifact '%s': %s",
                artifact_step.name,
                result.error,
            )

    def _execute_step(
        self,
        step_name: str,
        artifact_session: "Session",
        run_id: str,
        llm: "LLMClient",
        *,
        full: bool = False,
    ) -> dict[str, int]:
        """Execute a single step.

        Returns step-level stats.
        """
        from synix.services import records
        from synix.steps.aggregate import AggregateStep
        from synix.steps.fold import FoldStep
        from synix.steps.merge import MergeStep

        stats = {"input": 0, "output": 0, "skipped": 0, "errors": 0, "tokens": 0}

        # Handle sources
        if step_name in self._sources:
            source = self._sources[step_name]
            logger.info("Executing source: %s", step_name)

            for record in source.parse(run_id):
                stats["input"] += 1

                # Check if already exists
                existing = records.get_by_materialization_key(
                    artifact_session, record.materialization_key
                )
                if existing and not full:
                    stats["skipped"] += 1
                    continue

                # Save record
                records.create_record(
                    artifact_session,
                    content=record.content,
                    step_name=record.step_name,
                    materialization_key=record.materialization_key,
                    run_id=run_id,
                    metadata=record.metadata_,
                )
                stats["output"] += 1

            return stats

        # Handle transform/aggregate/fold/merge steps
        step = self._steps[step_name]
        logger.info("Executing step: %s (type=%s)", step_name, step.step_type)

        # Get inputs based on step type
        if isinstance(step, MergeStep):
            # MergeStep: gather from all sources
            input_records: list["Record"] = []
            for source_step in step.sources:
                source_records = records.get_records_by_step(artifact_session, source_step)
                input_records.extend(source_records)
            stats["input"] = len(input_records)

            # MergeStep produces a single output
            return self._execute_single_output_step(
                step, input_records, artifact_session, run_id, llm, stats, full
            )

        # For other steps, get inputs from single upstream
        from_step = step.from_
        if not from_step:
            msg = f"Step {step_name} has no from_ defined"
            raise ValueError(msg)

        input_records = records.get_records_by_step(artifact_session, from_step)
        stats["input"] = len(input_records)

        # Aggregate step: group inputs and process each group
        if isinstance(step, AggregateStep):
            groups = step.group_records(input_records)

            for group_key, group_records in groups.items():
                try:
                    version_hash = step.compute_version_hash()
                    mat_key = step.compute_materialization_key(group_records, version_hash)

                    # Check cache
                    existing = records.get_by_materialization_key(artifact_session, mat_key)
                    if existing and not full:
                        stats["skipped"] += 1
                        continue

                    # Execute step
                    output = step.execute(group_records, llm, run_id)
                    output.materialization_key = mat_key

                    # Save record with provenance
                    records.create_record(
                        artifact_session,
                        content=output.content,
                        step_name=output.step_name,
                        materialization_key=mat_key,
                        run_id=run_id,
                        sources=group_records,
                        metadata=output.metadata_,
                        audit=output.audit,
                    )

                    stats["output"] += 1
                    stats["tokens"] += output.audit.get("input_tokens", 0)
                    stats["tokens"] += output.audit.get("output_tokens", 0)

                except Exception as e:
                    logger.exception("Error processing group %s: %s", group_key, e)
                    stats["errors"] += 1

        # Fold step: process all inputs sequentially, single output
        elif isinstance(step, FoldStep):
            return self._execute_single_output_step(
                step, input_records, artifact_session, run_id, llm, stats, full
            )

        else:
            # Transform step: process each input individually
            for input_record in input_records:
                try:
                    version_hash = step.compute_version_hash()
                    mat_key = step.compute_materialization_key([input_record], version_hash)

                    # Check cache
                    existing = records.get_by_materialization_key(artifact_session, mat_key)
                    if existing and not full:
                        stats["skipped"] += 1
                        continue

                    # Execute step
                    output = step.execute([input_record], llm, run_id)
                    output.materialization_key = mat_key

                    # Save record with provenance
                    records.create_record(
                        artifact_session,
                        content=output.content,
                        step_name=output.step_name,
                        materialization_key=mat_key,
                        run_id=run_id,
                        sources=[input_record],
                        metadata=output.metadata_,
                        audit=output.audit,
                    )

                    stats["output"] += 1
                    stats["tokens"] += output.audit.get("input_tokens", 0)
                    stats["tokens"] += output.audit.get("output_tokens", 0)

                except Exception as e:
                    logger.exception("Error processing record %s: %s", input_record.id, e)
                    stats["errors"] += 1

        return stats

    def _execute_single_output_step(
        self,
        step: "Step",
        input_records: list["Record"],
        artifact_session: "Session",
        run_id: str,
        llm: "LLMClient",
        stats: dict[str, int],
        full: bool,
    ) -> dict[str, int]:
        """Execute a step that produces a single output from all inputs.

        Used for FoldStep and MergeStep.
        """
        from synix.services import records

        if not input_records:
            logger.warning("No inputs for step %s", step.name)
            return stats

        try:
            version_hash = step.compute_version_hash()
            mat_key = step.compute_materialization_key(input_records, version_hash)

            # Check cache
            existing = records.get_by_materialization_key(artifact_session, mat_key)
            if existing and not full:
                stats["skipped"] += 1
                return stats

            # Execute step
            output = step.execute(input_records, llm, run_id)
            output.materialization_key = mat_key

            # Save record with provenance
            records.create_record(
                artifact_session,
                content=output.content,
                step_name=output.step_name,
                materialization_key=mat_key,
                run_id=run_id,
                sources=input_records,
                metadata=output.metadata_,
                audit=output.audit,
            )

            stats["output"] += 1
            stats["tokens"] += output.audit.get("input_tokens", 0)
            stats["tokens"] += output.audit.get("output_tokens", 0)

        except Exception as e:
            logger.exception("Error executing step %s: %s", step.name, e)
            stats["errors"] += 1

        return stats

    def plan(self) -> PlanResult:
        """Show what would execute without actually running.

        Returns:
            PlanResult with step details and counts.
        """
        from synix.db.engine import get_artifact_session
        from synix.services import records
        from synix.steps.merge import MergeStep

        self._ensure_initialized()

        order = self._resolve_order()
        steps_info: list[dict[str, Any]] = []
        total_inputs = 0
        would_process = 0
        would_skip = 0

        with get_artifact_session(self.settings) as artifact_session:
            for step_name in order:
                info: dict[str, Any] = {"name": step_name}

                if step_name in self._sources:
                    source = self._sources[step_name]
                    info["type"] = "source"
                    info["format"] = source.format
                    info["file"] = str(source.file_path)

                    # Count existing records
                    existing = records.count_records_by_step(artifact_session, step_name)
                    info["existing"] = existing

                else:
                    step = self._steps[step_name]
                    info["type"] = step.step_type
                    info["version_hash"] = step.compute_version_hash()

                    # Get input count based on step type
                    if isinstance(step, MergeStep):
                        info["from"] = step.sources
                        input_count = 0
                        for source_step in step.sources:
                            inputs = records.get_records_by_step(artifact_session, source_step)
                            input_count += len(inputs)
                        info["inputs"] = input_count
                        total_inputs += input_count
                    elif step.from_:
                        info["from"] = step.from_
                        inputs = records.get_records_by_step(artifact_session, step.from_)
                        info["inputs"] = len(inputs)
                        total_inputs += len(inputs)

                    # Count existing outputs
                    existing = records.count_records_by_step(artifact_session, step_name)
                    info["existing"] = existing

                steps_info.append(info)

        return PlanResult(
            steps=steps_info,
            total_inputs=total_inputs,
            would_process=would_process,  # Would need full analysis
            would_skip=would_skip,
        )

    def search(
        self,
        query: str,
        *,
        step: str | None = None,
        limit: int = 10,
    ) -> list["SearchHit"]:
        """Search records using FTS.

        Args:
            query: Search query (FTS5 syntax supported).
            step: Optional step name to filter by.
            limit: Maximum results.

        Returns:
            List of SearchHit objects.
        """
        from synix.db.engine import get_artifact_session
        from synix.services.search import search_fts

        self._ensure_initialized()

        with get_artifact_session(self.settings) as artifact_session:
            return search_fts(artifact_session, query, step=step, limit=limit)

    def get(self, record_id: str) -> "Record | None":
        """Get a record by ID.

        Args:
            record_id: Record ID.

        Returns:
            Record or None.
        """
        from synix.db.engine import get_artifact_session
        from synix.services.records import get_record

        self._ensure_initialized()

        with get_artifact_session(self.settings) as artifact_session:
            return get_record(artifact_session, record_id)

    def get_records(self, step: str) -> list["Record"]:
        """Get all records for a step.

        Args:
            step: Step name.

        Returns:
            List of records.
        """
        from synix.db.engine import get_artifact_session
        from synix.services.records import get_records_by_step

        self._ensure_initialized()

        with get_artifact_session(self.settings) as artifact_session:
            return get_records_by_step(artifact_session, step)

    def get_sources(self, record_id: str) -> list["Record"]:
        """Get source records for a derived record.

        Args:
            record_id: ID of the derived record.

        Returns:
            List of source records.
        """
        from synix.db.engine import get_artifact_session
        from synix.services.records import get_record_sources

        self._ensure_initialized()

        with get_artifact_session(self.settings) as artifact_session:
            return get_record_sources(artifact_session, record_id)
