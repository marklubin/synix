"""Database engine setup for Synix.

Two-layer storage architecture:
- control_engine: Pipeline definitions, run tracking, step configs (control.db)
- artifact_engine: Records, provenance, FTS index (artifacts.db)
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from synix.config import Settings

# Lazy engine initialization - engines created on first use
_control_engine: Engine | None = None
_artifact_engine: Engine | None = None
_control_session_factory: sessionmaker[Session] | None = None
_artifact_session_factory: sessionmaker[Session] | None = None


def _enable_foreign_keys(dbapi_conn: object, connection_record: object) -> None:
    """Enable foreign key constraints for SQLite connections."""
    cursor = dbapi_conn.cursor()  # type: ignore[union-attr]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _create_engine_for_db(db_path: Path) -> Engine:
    """Create a SQLite engine with proper configuration."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        pool_pre_ping=True,
    )
    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


def get_control_engine(settings: "Settings | None" = None) -> Engine:
    """Get or create the control plane engine."""
    global _control_engine
    if _control_engine is None:
        if settings is None:
            from synix.config import get_settings

            settings = get_settings()
        settings.ensure_storage_dir()
        _control_engine = _create_engine_for_db(settings.control_db_path)
    return _control_engine


def get_artifact_engine(settings: "Settings | None" = None) -> Engine:
    """Get or create the data plane engine."""
    global _artifact_engine
    if _artifact_engine is None:
        if settings is None:
            from synix.config import get_settings

            settings = get_settings()
        settings.ensure_storage_dir()
        _artifact_engine = _create_engine_for_db(settings.artifact_db_path)
    return _artifact_engine


def get_control_session_factory(settings: "Settings | None" = None) -> sessionmaker[Session]:
    """Get or create the control plane session factory."""
    global _control_session_factory
    if _control_session_factory is None:
        engine = get_control_engine(settings)
        _control_session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return _control_session_factory


def get_artifact_session_factory(settings: "Settings | None" = None) -> sessionmaker[Session]:
    """Get or create the data plane session factory."""
    global _artifact_session_factory
    if _artifact_session_factory is None:
        engine = get_artifact_engine(settings)
        _artifact_session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return _artifact_session_factory


@contextmanager
def get_control_session(settings: "Settings | None" = None) -> Generator[Session, None, None]:
    """Yield a control plane database session."""
    factory = get_control_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_artifact_session(settings: "Settings | None" = None) -> Generator[Session, None, None]:
    """Yield a data plane database session."""
    factory = get_artifact_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_databases(settings: "Settings | None" = None) -> None:
    """Initialize both databases with their schemas."""
    from synix.db.artifacts import ArtifactBase, init_fts
    from synix.db.control import ControlBase

    # Create control plane tables
    control_engine = get_control_engine(settings)
    ControlBase.metadata.create_all(control_engine)

    # Create data plane tables
    artifact_engine = get_artifact_engine(settings)
    ArtifactBase.metadata.create_all(artifact_engine)

    # Initialize FTS for artifacts
    with artifact_engine.connect() as conn:
        init_fts(conn)
        conn.commit()


def reset_engines() -> None:
    """Reset all engine caches (useful for testing)."""
    global _control_engine, _artifact_engine
    global _control_session_factory, _artifact_session_factory

    if _control_engine is not None:
        _control_engine.dispose()
        _control_engine = None
    if _artifact_engine is not None:
        _artifact_engine.dispose()
        _artifact_engine = None
    _control_session_factory = None
    _artifact_session_factory = None


# Convenience aliases for backwards compatibility
control_engine = property(lambda self: get_control_engine())
artifact_engine = property(lambda self: get_artifact_engine())
