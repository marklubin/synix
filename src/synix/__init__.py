"""Synix — A build system for agent memory.

Re-export core models for backward compatibility.
"""

__version__ = "0.10.1"

from synix.core.models import (  # noqa: F401
    Artifact,
    FixerDecl,
    Layer,
    Pipeline,
    Projection,
    ProvenanceRecord,
    ValidatorDecl,
)
