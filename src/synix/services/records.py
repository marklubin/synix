"""Record CRUD and provenance operations (data plane)."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from synix.db.artifacts import Record, RecordSource


def create_record(
    session: Session,
    content: str,
    step_name: str,
    materialization_key: str,
    run_id: str | UUID,
    sources: list[Record] | None = None,
    metadata: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    branch: str = "main",
) -> Record:
    """Create a new record with provenance links.

    Args:
        session: Database session.
        content: Record content.
        step_name: Name of the step that produced this record.
        materialization_key: Unique cache key.
        run_id: ID of the run that created this record.
        sources: Optional list of source records (provenance).
        metadata: Optional metadata dict.
        audit: Optional audit info (prompt_hash, model, tokens).
        branch: Branch name (default: main).

    Returns:
        Created Record.
    """
    record = Record(
        id=str(uuid4()),
        content=content,
        content_fingerprint=Record.compute_fingerprint(content),
        step_name=step_name,
        branch=branch,
        materialization_key=materialization_key,
        run_id=str(run_id),
    )

    if metadata:
        record.metadata_ = metadata
    if audit:
        record.audit = audit

    session.add(record)

    # Create provenance links
    if sources:
        for i, source in enumerate(sources):
            link = RecordSource(
                record_id=record.id,
                source_id=source.id,
                source_order=i,
            )
            session.add(link)

    return record


def get_record(session: Session, record_id: str | UUID) -> Record | None:
    """Get a record by ID.

    Args:
        session: Database session.
        record_id: Record ID.

    Returns:
        Record or None if not found.
    """
    return session.get(Record, str(record_id))


def get_by_materialization_key(
    session: Session,
    materialization_key: str,
) -> Record | None:
    """Get a record by its materialization key.

    Used to check if an output already exists (cache hit).

    Args:
        session: Database session.
        materialization_key: Unique cache key.

    Returns:
        Record or None if not found.
    """
    stmt = select(Record).where(Record.materialization_key == materialization_key)
    return session.scalar(stmt)


def get_records_by_step(
    session: Session,
    step_name: str,
    branch: str = "main",
) -> list[Record]:
    """Get all records for a step.

    Args:
        session: Database session.
        step_name: Step name.
        branch: Branch name (default: main).

    Returns:
        List of records, ordered by created_at.
    """
    stmt = (
        select(Record)
        .where(Record.step_name == step_name, Record.branch == branch)
        .order_by(Record.created_at)
    )
    return list(session.scalars(stmt))


def get_record_sources(session: Session, record_id: str | UUID) -> list[Record]:
    """Get the source records for a derived record.

    Args:
        session: Database session.
        record_id: ID of the derived record.

    Returns:
        List of source records in order.
    """
    stmt = (
        select(Record)
        .join(RecordSource, RecordSource.source_id == Record.id)
        .where(RecordSource.record_id == str(record_id))
        .order_by(RecordSource.source_order)
    )
    return list(session.scalars(stmt))


def get_derived_records(session: Session, record_id: str | UUID) -> list[Record]:
    """Get records derived from a source record.

    Args:
        session: Database session.
        record_id: ID of the source record.

    Returns:
        List of derived records.
    """
    stmt = (
        select(Record)
        .join(RecordSource, RecordSource.record_id == Record.id)
        .where(RecordSource.source_id == str(record_id))
        .order_by(Record.created_at)
    )
    return list(session.scalars(stmt))


def count_records_by_step(
    session: Session,
    step_name: str,
    branch: str = "main",
) -> int:
    """Count records for a step.

    Args:
        session: Database session.
        step_name: Step name.
        branch: Branch name.

    Returns:
        Number of records.
    """
    from sqlalchemy import func

    stmt = (
        select(func.count())
        .select_from(Record)
        .where(Record.step_name == step_name, Record.branch == branch)
    )
    return session.scalar(stmt) or 0


def delete_records_by_step(
    session: Session,
    step_name: str,
    branch: str = "main",
) -> int:
    """Delete all records for a step.

    Used for full reprocessing.

    Args:
        session: Database session.
        step_name: Step name.
        branch: Branch name.

    Returns:
        Number of records deleted.
    """
    from sqlalchemy import delete

    stmt = delete(Record).where(Record.step_name == step_name, Record.branch == branch)
    result = session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]
