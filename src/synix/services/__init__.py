"""Service layer for Synix operations.

Control plane services (control.db):
- pipelines: Pipeline CRUD
- runs: Run tracking

Data plane services (artifacts.db):
- records: Record CRUD + provenance
- search: FTS search
"""

from synix.services import pipelines, records, runs, search

__all__ = [
    "pipelines",
    "records",
    "runs",
    "search",
]
