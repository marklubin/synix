"""Database models and engines for Synix.

Two-layer storage architecture:
- Control plane (control.db): Pipeline definitions, run tracking, step configs
- Data plane (artifacts.db): Records, provenance, FTS index
"""

from synix.db.artifacts import ArtifactBase, Record, RecordSource
from synix.db.control import Branch, ControlBase, PipelineState, Run, StepConfig
from synix.db.engine import (
    get_artifact_engine,
    get_artifact_session,
    get_control_engine,
    get_control_session,
    init_databases,
)

__all__ = [
    "ArtifactBase",
    "Branch",
    "ControlBase",
    "PipelineState",
    "Record",
    "RecordSource",
    "Run",
    "StepConfig",
    "get_artifact_engine",
    "get_artifact_session",
    "get_control_engine",
    "get_control_session",
    "init_databases",
]
