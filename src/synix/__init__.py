"""Synix — A build system for agent memory.

Re-export core models for backward compatibility.
"""

__version__ = "0.9.3"

from synix.core.models import (  # noqa: F401
    Artifact,
    FixerDecl,
    Layer,
    Pipeline,
    Projection,
    ProvenanceRecord,
    ValidatorDecl,
)
