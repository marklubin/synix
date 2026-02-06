"""Control plane database models for Synix.

These models track pipeline infrastructure:
- PipelineState: Pipeline definitions
- StepConfig: Step configurations with version hashes
- Run: Execution tracking
- Branch: Branch metadata (stub for v1.0)
"""

import json
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class ControlBase(DeclarativeBase):
    """Base class for control plane models."""

    type_annotation_map: ClassVar[dict[type, Any]] = {}


class PipelineState(ControlBase):
    """Pipeline definition storage."""

    __tablename__ = "pipeline_states"

    name: Mapped[str] = mapped_column(String(256), primary_key=True)
    agent: Mapped[str] = mapped_column(String(256), nullable=False)
    definition_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    step_configs: Mapped[list["StepConfig"]] = relationship(
        "StepConfig", back_populates="pipeline", cascade="all, delete-orphan"
    )
    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="pipeline", cascade="all, delete-orphan"
    )

    @property
    def definition(self) -> dict[str, Any]:
        """Get deserialized pipeline definition."""
        return json.loads(self.definition_json)  # type: ignore[no-any-return]

    @definition.setter
    def definition(self, value: dict[str, Any]) -> None:
        """Set serialized pipeline definition."""
        self.definition_json = json.dumps(value)


class StepConfig(ControlBase):
    """Step configuration materialized from Pipeline definition."""

    __tablename__ = "step_configs"

    id: Mapped[UUID] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    pipeline_name: Mapped[str] = mapped_column(
        String(256), ForeignKey("pipeline_states.name", ondelete="CASCADE"), nullable=False
    )
    step_name: Mapped[str] = mapped_column(String(256), nullable=False)
    step_type: Mapped[str] = mapped_column(String(64), nullable=False)  # source/transform/aggregate
    from_step: Mapped[str | None] = mapped_column(String(256), nullable=True)  # upstream dependency
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # hash of prompt + config
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    pipeline: Mapped[PipelineState] = relationship("PipelineState", back_populates="step_configs")

    @property
    def config(self) -> dict[str, Any]:
        """Get deserialized step config."""
        return json.loads(self.config_json)  # type: ignore[no-any-return]

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        """Set serialized step config."""
        self.config_json = json.dumps(value)

    __table_args__ = (
        UniqueConstraint("pipeline_name", "step_name", name="uq_step_config_pipeline_step"),
        Index("idx_step_configs_pipeline", "pipeline_name"),
    )


class Run(ControlBase):
    """Execution tracking for pipeline runs."""

    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    pipeline_name: Mapped[str] = mapped_column(
        String(256), ForeignKey("pipeline_states.name", ondelete="CASCADE"), nullable=False
    )
    branch: Mapped[str] = mapped_column(String(256), nullable=False, default="main")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending/running/completed/failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stats_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    pipeline: Mapped[PipelineState] = relationship("PipelineState", back_populates="runs")

    @property
    def stats(self) -> dict[str, Any]:
        """Get deserialized run stats."""
        return json.loads(self.stats_json)  # type: ignore[no-any-return]

    @stats.setter
    def stats(self, value: dict[str, Any]) -> None:
        """Set serialized run stats."""
        self.stats_json = json.dumps(value)

    __table_args__ = (
        Index("idx_runs_pipeline", "pipeline_name"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_created_at", "created_at"),
    )


class Branch(ControlBase):
    """Branch metadata (stub for v1.0, always 'main' in v0.1)."""

    __tablename__ = "branches"

    name: Mapped[str] = mapped_column(String(256), primary_key=True)
    pipeline_name: Mapped[str] = mapped_column(
        String(256), ForeignKey("pipeline_states.name", ondelete="CASCADE"), nullable=False
    )
    parent_branch: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_branches_pipeline", "pipeline_name"),)
