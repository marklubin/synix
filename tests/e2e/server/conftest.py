"""E2E server test fixtures — creates a real Starlette test client for the knowledge server."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from synix.server.api import api_routes
from synix.server.config import AutoBuildConfig, BucketConfig, ServerConfig
from synix.server.mcp_tools import _state, server_mcp

TOY_PIPELINE_DIR = Path(__file__).parent / "toy-pipeline"


@pytest.fixture
def server_project(tmp_path, mock_llm):
    """Create a full synix project with toy pipeline, ready for builds.

    Returns (project_dir, config) tuple.
    """
    import synix

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Copy pipeline
    shutil.copy(TOY_PIPELINE_DIR / "pipeline.py", project_dir / "pipeline.py")

    # Create source and bucket directories
    sources = project_dir / "sources"
    (sources / "documents").mkdir(parents=True)
    (sources / "sessions").mkdir(parents=True)
    (sources / "reports").mkdir(parents=True)

    # Also create the pipeline's source_dir (./sources is what the pipeline references,
    # but we use the "notes" Source which defaults to the source_dir)
    # The pipeline expects source_dir at the project level
    # Note: ingest writes to bucket dirs; the pipeline reads from source_dir.
    # For this test, we'll make the pipeline's source_dir point to sources/documents
    # so ingested documents are visible to the pipeline.

    # Init synix project
    synix.init(str(project_dir))

    config = ServerConfig(
        project_dir=str(project_dir),
        pipeline_path="pipeline.py",
        mcp_port=8200,
        viewer_port=9471,
        buckets=[
            BucketConfig(
                name="documents",
                dir="sources/documents",
                patterns=["**/*.md", "**/*.txt"],
                description="Notes and documents",
            ),
            BucketConfig(
                name="sessions",
                dir="sources/sessions",
                patterns=["**/*.jsonl", "**/*.jsonl.gz"],
                description="Session transcripts",
            ),
            BucketConfig(
                name="reports",
                dir="sources/reports",
                patterns=["**/*.md"],
                description="Automated reports",
            ),
        ],
        auto_build=AutoBuildConfig(enabled=False),
    )

    return project_dir, config


@pytest.fixture
def server_app(server_project):
    """Create the Starlette app with MCP + REST routes, state initialized."""
    import synix

    project_dir, config = server_project

    project = synix.open_project(str(project_dir))
    pipeline_path = Path(project_dir) / config.pipeline_path
    if pipeline_path.exists():
        project.load_pipeline(str(pipeline_path))

    # Set global state (same as serve.py does)
    _state["project"] = project
    _state["config"] = config

    app = server_mcp.streamable_http_app()
    app.routes.extend(api_routes)

    yield app

    # Cleanup
    _state["project"] = None
    _state["config"] = None


@pytest.fixture
def client(server_app):
    """Starlette test client for the knowledge server."""
    return TestClient(server_app)


@pytest.fixture
def built_server(server_project, mock_llm):
    """Server with data ingested, built, and released — ready for search.

    Ingests 3 documents, runs build + release, returns (client, project_dir, config).
    """
    import synix

    project_dir, config = server_project

    # Ingest test documents into the pipeline's source directory
    # The toy pipeline reads from ./sources (relative to project_dir)
    notes_dir = Path(project_dir) / "sources"
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / "note-ai.md").write_text(
        "Today I explored machine learning fundamentals. "
        "We discussed neural networks, gradient descent, and backpropagation. "
        "Key insight: transformers revolutionized NLP."
    )
    (notes_dir / "note-systems.md").write_text(
        "Reviewed distributed systems concepts. "
        "Covered consensus algorithms, Raft protocol, and CAP theorem. "
        "Agent memory requires strong consistency for lifecycle state."
    )
    (notes_dir / "note-synix.md").write_text(
        "Synix architecture session: tier model with compression, "
        "division, and renewal. Memory pipeline processes raw conversations "
        "into searchable, hierarchical artifacts with full provenance."
    )

    # Also put them in the documents bucket for ingest tests
    docs_dir = Path(project_dir) / "sources" / "documents"
    for f in notes_dir.glob("*.md"):
        shutil.copy(f, docs_dir / f.name)

    # Build and release via SDK (no build internals)
    project = synix.open_project(str(project_dir))
    project.load_pipeline(str(Path(project_dir) / "pipeline.py"))
    result = project.build()
    assert result.built > 0, f"Build produced no artifacts: {result}"
    project.release_to("local")

    # Re-open to pick up release
    project = synix.open_project(str(project_dir))

    # Set global state
    _state["project"] = project
    _state["config"] = config

    app = server_mcp.streamable_http_app()
    app.routes.extend(api_routes)

    yield TestClient(app), project_dir, config

    _state["project"] = None
    _state["config"] = None
