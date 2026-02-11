"""Structured logging and verbosity levels for Synix builds."""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from threading import Lock
from typing import Any


class Verbosity(IntEnum):
    """Verbosity levels for console output."""

    DEFAULT = 0  # Summary table only
    VERBOSE = 1  # + per-layer progress, per-artifact status
    DEBUG = 2  # + LLM request/response details, timing


@dataclass
class StepLog:
    """Per-layer build statistics."""

    name: str
    llm_calls: int = 0
    cache_hits: int = 0
    rebuilt_ids: list[str] = field(default_factory=list)
    cached_ids: list[str] = field(default_factory=list)
    time_seconds: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "rebuilt_ids": list(self.rebuilt_ids),
            "cached_ids": list(self.cached_ids),
            "time_seconds": self.time_seconds,
            "tokens_used": self.tokens_used,
        }


@dataclass
class RunLog:
    """Structured log of a complete pipeline run.

    Serializable to dict for use with tests/helpers/assertions.py.
    The dict format is::

        {
            "steps": {
                "episodes": {
                    "llm_calls": 8,
                    "cache_hits": 0,
                    "rebuilt_ids": ["ep-conv001", ...],
                    "cached_ids": [],
                    "time_seconds": 2.3,
                    "tokens_used": 1200,
                },
                ...
            },
            "total_llm_calls": 11,
            "total_cache_hits": 0,
            "total_time": 5.4,
            "total_tokens": 3200,
            "total_cost_estimate": 0.05,
        }
    """

    run_id: str = ""
    steps: dict[str, StepLog] = field(default_factory=dict)
    total_time: float = 0.0
    total_llm_calls: int = 0
    total_cache_hits: int = 0
    total_tokens: int = 0
    total_cost_estimate: float = 0.0

    def get_or_create_step(self, name: str) -> StepLog:
        """Get existing step log or create a new one."""
        if name not in self.steps:
            self.steps[name] = StepLog(name=name)
        return self.steps[name]

    def finalize(self) -> None:
        """Compute totals from step data."""
        self.total_llm_calls = sum(s.llm_calls for s in self.steps.values())
        self.total_cache_hits = sum(s.cache_hits for s in self.steps.values())
        self.total_tokens = sum(s.tokens_used for s in self.steps.values())
        self.total_cost_estimate = _estimate_cost(self.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict matching the format expected by assertion helpers."""
        return {
            "run_id": self.run_id,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "total_llm_calls": self.total_llm_calls,
            "total_cache_hits": self.total_cache_hits,
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "total_cost_estimate": self.total_cost_estimate,
        }


@dataclass
class RunSummary:
    """Human-readable summary of a pipeline run."""

    total_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_estimate: float = 0.0
    cache_hit_rate: float = 0.0
    built: int = 0
    cached: int = 0


class SynixLogger:
    """Structured logger for Synix pipeline runs.

    Writes JSONL log files to build_dir/logs/ and optionally emits
    console output via Rich based on verbosity level.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.DEFAULT,
        build_dir: Path | None = None,
        progress: Any | None = None,
    ):
        self.verbosity = verbosity
        self.build_dir = build_dir
        self.progress = progress
        self.run_log = RunLog(
            run_id=datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        )
        self._lock = Lock()
        self._log_file = None
        self._log_path: Path | None = None
        self._current_step: str | None = None
        self._step_start: float = 0.0

        # Open JSONL log file if build_dir exists
        if build_dir is not None:
            logs_dir = build_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = logs_dir / f"{self.run_log.run_id}.jsonl"
            self._log_file = open(self._log_path, "a")

    def _write_event(self, event: dict[str, Any]) -> None:
        """Write a JSON event to the JSONL log file (thread-safe)."""
        with self._lock:
            if self._log_file is not None:
                event["timestamp"] = datetime.now(_dt.UTC).isoformat()
                self._log_file.write(json.dumps(event) + "\n")
                self._log_file.flush()

    def _console_print(self, message: str, min_verbosity: Verbosity) -> None:
        """Print to console if verbosity is high enough.

        Suppressed when a live progress display is active.
        """
        if self.progress:
            return  # Live display handles output
        if self.verbosity >= min_verbosity:
            try:
                from rich.console import Console

                console = Console()
                console.print(message)
            except ImportError:
                print(message)

    # -- Layer events --

    def layer_start(self, layer_name: str, level: int) -> None:
        """Log the start of a layer build."""
        self._current_step = layer_name
        self._step_start = time.time()
        self.run_log.get_or_create_step(layer_name)

        self._write_event(
            {
                "event": "layer_start",
                "layer": layer_name,
                "level": level,
            }
        )

        if self.progress:
            self.progress.layer_start(layer_name, level)

        self._console_print(
            f"  [bold]Building layer:[/bold] {layer_name} (level {level})",
            Verbosity.VERBOSE,
        )

    def layer_finish(self, layer_name: str, built: int, cached: int) -> None:
        """Log the completion of a layer build."""
        elapsed = time.time() - self._step_start
        step = self.run_log.get_or_create_step(layer_name)
        step.time_seconds = elapsed

        self._write_event(
            {
                "event": "layer_finish",
                "layer": layer_name,
                "built": built,
                "cached": cached,
                "time_seconds": round(elapsed, 3),
            }
        )

        if self.progress:
            self.progress.layer_finish(layer_name, built, cached)

        self._console_print(
            f"    {layer_name}: {built} built, {cached} cached ({elapsed:.1f}s)",
            Verbosity.VERBOSE,
        )
        self._current_step = None

    # -- Artifact events --

    def artifact_built(self, layer_name: str, artifact_id: str) -> None:
        """Log that an artifact was built (not cached)."""
        with self._lock:
            step = self.run_log.get_or_create_step(layer_name)
            step.rebuilt_ids.append(artifact_id)

        self._write_event(
            {
                "event": "artifact_built",
                "layer": layer_name,
                "artifact_id": artifact_id,
            }
        )

        self._console_print(
            f"      [green]+[/green] {artifact_id}",
            Verbosity.VERBOSE,
        )

    def artifact_cached(self, layer_name: str, artifact_id: str) -> None:
        """Log that an artifact was found in cache."""
        with self._lock:
            step = self.run_log.get_or_create_step(layer_name)
            step.cache_hits += 1
            step.cached_ids.append(artifact_id)

        self._write_event(
            {
                "event": "artifact_cached",
                "layer": layer_name,
                "artifact_id": artifact_id,
            }
        )

        if self.progress:
            self.progress.artifact_cached(artifact_id)

        self._console_print(
            f"      [cyan]=[/cyan] {artifact_id} (cached)",
            Verbosity.VERBOSE,
        )

    # -- LLM call events --

    def llm_call_start(
        self,
        layer_name: str,
        artifact_desc: str,
        model: str,
    ) -> float:
        """Log the start of an LLM call. Returns start time for pairing with llm_call_finish."""
        start = time.time()

        self._write_event(
            {
                "event": "llm_call_start",
                "layer": layer_name,
                "artifact_desc": artifact_desc,
                "model": model,
            }
        )

        if self.progress:
            self.progress.artifact_start(artifact_desc)

        self._console_print(
            f"        [dim]LLM call: {artifact_desc} ({model})[/dim]",
            Verbosity.DEBUG,
        )

        return start

    def llm_call_finish(
        self,
        layer_name: str,
        artifact_desc: str,
        start_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Log the completion of an LLM call."""
        elapsed = time.time() - start_time
        total_tokens = input_tokens + output_tokens

        with self._lock:
            step = self.run_log.get_or_create_step(layer_name)
            step.llm_calls += 1
            step.tokens_used += total_tokens

        self._write_event(
            {
                "event": "llm_call_finish",
                "layer": layer_name,
                "artifact_desc": artifact_desc,
                "duration_seconds": round(elapsed, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        if self.progress:
            self.progress.artifact_finish(artifact_desc, elapsed)

        self._console_print(
            f"        [dim]  -> {elapsed:.1f}s, {input_tokens}in/{output_tokens}out tokens[/dim]",
            Verbosity.DEBUG,
        )

    # -- Projection events --

    def projection_start(self, name: str, projection_type: str, triggered_by: str | None = None) -> None:
        """Log the start of a projection materialization."""
        event: dict[str, Any] = {
            "event": "projection_start",
            "projection": name,
            "projection_type": projection_type,
        }
        if triggered_by:
            event["triggered_by"] = triggered_by

        self._write_event(event)

        if self.progress:
            self.progress.projection_start(name, triggered_by=triggered_by)

        self._console_print(
            f"  [bold]Materializing projection:[/bold] {name} ({projection_type})",
            Verbosity.VERBOSE,
        )

    def projection_finish(self, name: str, triggered_by: str | None = None) -> None:
        """Log the completion of a projection materialization."""
        event: dict[str, Any] = {
            "event": "projection_finish",
            "projection": name,
        }
        if triggered_by:
            event["triggered_by"] = triggered_by

        self._write_event(event)

        if self.progress:
            self.progress.projection_finish(name, triggered_by=triggered_by)

        self._console_print(
            f"    [green]✓[/green] {name} materialized",
            Verbosity.VERBOSE,
        )

    # -- Embedding events --

    def embedding_start(self, count: int, provider: str) -> None:
        """Log the start of embedding generation."""
        self._write_event(
            {
                "event": "embedding_start",
                "count": count,
                "provider": provider,
            }
        )

        if self.progress:
            self.progress.embedding_start(count, provider)

        self._console_print(
            f"      [dim]Generating {count} embeddings ({provider})[/dim]",
            Verbosity.VERBOSE,
        )

    def embedding_progress(self, completed: int, total: int) -> None:
        """Log embedding generation progress."""
        if self.progress:
            self.progress.embedding_progress(completed, total)

    def embedding_finish(self, count: int, cached: int, generated: int) -> None:
        """Log the completion of embedding generation."""
        self._write_event(
            {
                "event": "embedding_finish",
                "count": count,
                "cached": cached,
                "generated": generated,
            }
        )

        if self.progress:
            self.progress.embedding_finish(count, cached, generated)

        self._console_print(
            f"      [dim]{count} embeddings ({cached} cached, {generated} generated)[/dim]",
            Verbosity.VERBOSE,
        )

    def projection_cached(self, name: str, triggered_by: str | None = None) -> None:
        """Log that a projection was skipped (cached)."""
        event: dict[str, Any] = {
            "event": "projection_cached",
            "projection": name,
        }
        if triggered_by:
            event["triggered_by"] = triggered_by

        self._write_event(event)

        if self.progress:
            self.progress.projection_cached(name, triggered_by=triggered_by)

        self._console_print(
            f"    [cyan]=[/cyan] {name} (cached)",
            Verbosity.VERBOSE,
        )

    # -- Run lifecycle --

    def run_start(self, pipeline_name: str, layer_count: int) -> None:
        """Log the start of a pipeline run."""
        self._write_event(
            {
                "event": "run_start",
                "pipeline": pipeline_name,
                "layer_count": layer_count,
            }
        )

    def run_finish(self, total_time: float) -> None:
        """Log the completion of a pipeline run and finalize stats."""
        self.run_log.total_time = total_time
        self.run_log.finalize()

        self._write_event(
            {
                "event": "run_finish",
                "total_time": round(total_time, 3),
                "total_llm_calls": self.run_log.total_llm_calls,
                "total_tokens": self.run_log.total_tokens,
                "total_cost_estimate": round(self.run_log.total_cost_estimate, 4),
            }
        )

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def get_summary(self) -> RunSummary:
        """Get a human-readable summary of the run."""
        total_artifacts = sum(len(s.rebuilt_ids) + len(s.cached_ids) for s in self.run_log.steps.values())
        cache_hit_rate = 0.0
        if total_artifacts > 0:
            cache_hit_rate = self.run_log.total_cache_hits / total_artifacts

        return RunSummary(
            total_time=self.run_log.total_time,
            total_llm_calls=self.run_log.total_llm_calls,
            total_tokens=self.run_log.total_tokens,
            total_cost_estimate=self.run_log.total_cost_estimate,
            cache_hit_rate=cache_hit_rate,
            built=sum(len(s.rebuilt_ids) for s in self.run_log.steps.values()),
            cached=self.run_log.total_cache_hits,
        )

    def close(self) -> None:
        """Close the log file if open."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def _estimate_cost(total_tokens: int) -> float:
    """Rough cost estimate based on Claude Sonnet pricing.

    Sonnet ~$3/1M input + $15/1M output. We approximate with an average
    of $6/1M tokens since we don't track input/output separately at the
    aggregate level.
    """
    return total_tokens * 6.0 / 1_000_000
