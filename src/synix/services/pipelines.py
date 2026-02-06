"""Pipeline CRUD operations (control plane)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from synix.db.control import PipelineState, StepConfig


def save_pipeline(
    session: Session,
    name: str,
    agent: str,
    definition: dict[str, Any],
) -> PipelineState:
    """Save or update a pipeline definition.

    Args:
        session: Database session.
        name: Pipeline name (primary key).
        agent: Agent name.
        definition: Serialized step graph.

    Returns:
        Created or updated PipelineState.
    """
    pipeline = session.get(PipelineState, name)
    if pipeline is None:
        pipeline = PipelineState(
            name=name,
            agent=agent,
        )
        session.add(pipeline)

    pipeline.agent = agent
    pipeline.definition = definition
    pipeline.updated_at = datetime.now()

    return pipeline


def get_pipeline(session: Session, name: str) -> PipelineState | None:
    """Get a pipeline by name.

    Args:
        session: Database session.
        name: Pipeline name.

    Returns:
        PipelineState or None if not found.
    """
    return session.get(PipelineState, name)


def list_pipelines(session: Session) -> list[PipelineState]:
    """List all pipelines.

    Args:
        session: Database session.

    Returns:
        List of all PipelineState objects.
    """
    stmt = select(PipelineState).order_by(PipelineState.updated_at.desc())
    return list(session.scalars(stmt))


def delete_pipeline(session: Session, name: str) -> bool:
    """Delete a pipeline and all related data.

    Args:
        session: Database session.
        name: Pipeline name.

    Returns:
        True if deleted, False if not found.
    """
    pipeline = session.get(PipelineState, name)
    if pipeline is None:
        return False
    session.delete(pipeline)
    return True


def save_step_config(
    session: Session,
    pipeline_name: str,
    step_name: str,
    step_type: str,
    from_step: str | None,
    version_hash: str,
    config: dict[str, Any],
) -> StepConfig:
    """Save or update a step configuration.

    Args:
        session: Database session.
        pipeline_name: Pipeline this step belongs to.
        step_name: Step name.
        step_type: Type (source, transform, aggregate).
        from_step: Upstream step name.
        version_hash: Hash of prompt + config.
        config: Step-specific configuration.

    Returns:
        Created or updated StepConfig.
    """
    # Look for existing
    stmt = select(StepConfig).where(
        StepConfig.pipeline_name == pipeline_name,
        StepConfig.step_name == step_name,
    )
    step_config = session.scalar(stmt)

    if step_config is None:
        step_config = StepConfig(
            pipeline_name=pipeline_name,
            step_name=step_name,
            step_type=step_type,
            from_step=from_step,
            version_hash=version_hash,
        )
        session.add(step_config)
    else:
        step_config.step_type = step_type
        step_config.from_step = from_step
        step_config.version_hash = version_hash
        step_config.updated_at = datetime.now()

    step_config.config = config

    return step_config


def get_step_config(
    session: Session,
    pipeline_name: str,
    step_name: str,
) -> StepConfig | None:
    """Get a step configuration.

    Args:
        session: Database session.
        pipeline_name: Pipeline name.
        step_name: Step name.

    Returns:
        StepConfig or None if not found.
    """
    stmt = select(StepConfig).where(
        StepConfig.pipeline_name == pipeline_name,
        StepConfig.step_name == step_name,
    )
    return session.scalar(stmt)


def get_step_configs(session: Session, pipeline_name: str) -> list[StepConfig]:
    """Get all step configs for a pipeline.

    Args:
        session: Database session.
        pipeline_name: Pipeline name.

    Returns:
        List of StepConfig objects.
    """
    stmt = select(StepConfig).where(StepConfig.pipeline_name == pipeline_name)
    return list(session.scalars(stmt))
