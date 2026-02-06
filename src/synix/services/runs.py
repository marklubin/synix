"""Run tracking operations (control plane)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from synix.db.control import Run


def create_run(session: Session, pipeline_name: str, branch: str = "main") -> Run:
    """Create a new run.

    Args:
        session: Database session.
        pipeline_name: Name of the pipeline being run.
        branch: Branch name (default: main).

    Returns:
        Created Run with status 'pending'.
    """
    run = Run(
        id=str(uuid4()),
        pipeline_name=pipeline_name,
        branch=branch,
        status="pending",
    )
    run.stats = {"input": 0, "output": 0, "skipped": 0, "errors": 0, "tokens": 0}
    session.add(run)
    return run


def start_run(session: Session, run_id: str | UUID) -> Run:
    """Mark a run as started.

    Args:
        session: Database session.
        run_id: Run ID.

    Returns:
        Updated Run with status 'running'.
    """
    run = session.get(Run, str(run_id))
    if run is None:
        msg = f"Run {run_id} not found"
        raise ValueError(msg)

    run.status = "running"
    run.started_at = datetime.now()
    return run


def complete_run(session: Session, run_id: str | UUID, stats: dict[str, Any]) -> Run:
    """Mark a run as completed.

    Args:
        session: Database session.
        run_id: Run ID.
        stats: Run statistics (input, output, skipped, errors, tokens).

    Returns:
        Updated Run with status 'completed'.
    """
    run = session.get(Run, str(run_id))
    if run is None:
        msg = f"Run {run_id} not found"
        raise ValueError(msg)

    run.status = "completed"
    run.completed_at = datetime.now()
    run.stats = stats
    return run


def fail_run(session: Session, run_id: str | UUID, error: str) -> Run:
    """Mark a run as failed.

    Args:
        session: Database session.
        run_id: Run ID.
        error: Error message.

    Returns:
        Updated Run with status 'failed'.
    """
    run = session.get(Run, str(run_id))
    if run is None:
        msg = f"Run {run_id} not found"
        raise ValueError(msg)

    run.status = "failed"
    run.completed_at = datetime.now()
    run.error_message = error
    return run


def get_run(session: Session, run_id: str | UUID) -> Run | None:
    """Get a run by ID.

    Args:
        session: Database session.
        run_id: Run ID.

    Returns:
        Run or None if not found.
    """
    return session.get(Run, str(run_id))


def get_runs(
    session: Session,
    pipeline_name: str,
    limit: int = 10,
    status: str | None = None,
) -> list[Run]:
    """Get runs for a pipeline.

    Args:
        session: Database session.
        pipeline_name: Pipeline name.
        limit: Maximum runs to return.
        status: Optional status filter.

    Returns:
        List of Run objects, newest first.
    """
    stmt = (
        select(Run)
        .where(Run.pipeline_name == pipeline_name)
        .order_by(Run.created_at.desc())
        .limit(limit)
    )

    if status:
        stmt = stmt.where(Run.status == status)

    return list(session.scalars(stmt))


def get_latest_run(session: Session, pipeline_name: str) -> Run | None:
    """Get the most recent run for a pipeline.

    Args:
        session: Database session.
        pipeline_name: Pipeline name.

    Returns:
        Most recent Run or None.
    """
    stmt = (
        select(Run)
        .where(Run.pipeline_name == pipeline_name)
        .order_by(Run.created_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)
