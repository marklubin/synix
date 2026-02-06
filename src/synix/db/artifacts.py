"""Data plane database models for Synix.

These models store the actual content artifacts:
- Record: Text content with provenance and metadata
- RecordSource: Provenance links between records

FTS5 virtual table provides full-text search on record content.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    Connection,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class ArtifactBase(DeclarativeBase):
    """Base class for data plane models."""

    type_annotation_map: ClassVar[dict[type, Any]] = {}


class Record(ArtifactBase):
    """The actual content artifacts produced by pipeline steps."""

    __tablename__ = "records"

    id: Mapped[UUID] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256
    step_name: Mapped[str] = mapped_column(String(256), nullable=False)
    branch: Mapped[str] = mapped_column(String(256), nullable=False, default="main")
    materialization_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    run_id: Mapped[UUID] = mapped_column(String(36), nullable=False)  # Links to control.Run
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    audit_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    # Relationships
    sources: Mapped[list["RecordSource"]] = relationship(
        "RecordSource",
        foreign_keys="RecordSource.record_id",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    derived_records: Mapped[list["RecordSource"]] = relationship(
        "RecordSource",
        foreign_keys="RecordSource.source_id",
        back_populates="source",
    )

    @property
    def metadata_(self) -> dict[str, Any]:
        """Get deserialized metadata."""
        return json.loads(self.metadata_json)  # type: ignore[no-any-return]

    @metadata_.setter
    def metadata_(self, value: dict[str, Any]) -> None:
        """Set serialized metadata."""
        self.metadata_json = json.dumps(value)

    @property
    def audit(self) -> dict[str, Any]:
        """Get deserialized audit info."""
        return json.loads(self.audit_json)  # type: ignore[no-any-return]

    @audit.setter
    def audit(self, value: dict[str, Any]) -> None:
        """Set serialized audit info."""
        self.audit_json = json.dumps(value)

    @staticmethod
    def compute_fingerprint(content: str) -> str:
        """Compute SHA-256 fingerprint of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    __table_args__ = (
        Index("idx_records_step", "step_name"),
        Index("idx_records_branch", "branch"),
        Index("idx_records_run", "run_id"),
        Index("idx_records_created_at", "created_at"),
        Index("idx_records_fingerprint", "content_fingerprint"),
    )


class RecordSource(ArtifactBase):
    """Provenance links between records."""

    __tablename__ = "record_sources"

    id: Mapped[UUID] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    record_id: Mapped[UUID] = mapped_column(
        String(36), ForeignKey("records.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[UUID] = mapped_column(
        String(36), ForeignKey("records.id", ondelete="CASCADE"), nullable=False
    )
    source_order: Mapped[int] = mapped_column(default=0)  # For ordered inputs
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    record: Mapped[Record] = relationship(
        "Record", foreign_keys=[record_id], back_populates="sources"
    )
    source: Mapped[Record] = relationship(
        "Record", foreign_keys=[source_id], back_populates="derived_records"
    )

    __table_args__ = (
        UniqueConstraint("record_id", "source_id", name="uq_record_source"),
        Index("idx_record_sources_record", "record_id"),
        Index("idx_record_sources_source", "source_id"),
    )


# FTS5 setup SQL
# Using external content table with explicit rowid
FTS_CREATE_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS record_fts USING fts5(
    content,
    step_name
);
"""

# Separate table to map FTS rowids to record IDs
FTS_MAP_TABLE = """
CREATE TABLE IF NOT EXISTS record_fts_map (
    rowid INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL
);
"""

FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS record_fts_insert AFTER INSERT ON records BEGIN
    INSERT INTO record_fts(content, step_name)
    VALUES (NEW.content, NEW.step_name);
    INSERT INTO record_fts_map(rowid, record_id)
    VALUES (last_insert_rowid(), NEW.id);
END;
"""

FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS record_fts_update AFTER UPDATE ON records BEGIN
    DELETE FROM record_fts
      WHERE rowid = (SELECT rowid FROM record_fts_map WHERE record_id = OLD.id);
    DELETE FROM record_fts_map WHERE record_id = OLD.id;
    INSERT INTO record_fts(content, step_name)
    VALUES (NEW.content, NEW.step_name);
    INSERT INTO record_fts_map(rowid, record_id)
    VALUES (last_insert_rowid(), NEW.id);
END;
"""

FTS_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS record_fts_delete AFTER DELETE ON records BEGIN
    DELETE FROM record_fts
      WHERE rowid = (SELECT rowid FROM record_fts_map WHERE record_id = OLD.id);
    DELETE FROM record_fts_map WHERE record_id = OLD.id;
END;
"""


def init_fts(conn: Connection) -> None:
    """Initialize FTS5 virtual table and triggers."""
    conn.execute(text(FTS_CREATE_TABLE))
    conn.execute(text(FTS_MAP_TABLE))
    conn.execute(text(FTS_INSERT_TRIGGER))
    conn.execute(text(FTS_UPDATE_TRIGGER))
    conn.execute(text(FTS_DELETE_TRIGGER))


def rebuild_fts(conn: Connection) -> None:
    """Rebuild FTS index from scratch."""
    # Delete all FTS data
    conn.execute(text("DELETE FROM record_fts"))
    # Repopulate from records table
    conn.execute(
        text("""
        INSERT INTO record_fts(rowid, record_id, content, step_name)
        SELECT rowid, id, content, step_name FROM records
    """)
    )
