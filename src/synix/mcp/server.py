"""Synix MCP server — full pipeline control via stdin/stdout.

Exposes the complete SDK as MCP tools so AI agents can manage
synix pipelines, build memory, and search — all via structured API.

Usage:
    python -m synix.mcp                             # stdio transport
    SYNIX_PROJECT=./my-project python -m synix.mcp   # auto-open project

Configure in MCP client settings:
    {
        "mcpServers": {
            "synix": {
                "command": "uvx",
                "args": ["--from", "synix[mcp]", "python", "-m", "synix.mcp"],
                "env": {"SYNIX_PROJECT": "/path/to/project"}
            }
        }
    }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "synix",
    instructions=(
        "Synix — a build system for agent memory. "
        "Declarative pipelines define how raw conversations become searchable, "
        "hierarchical memory with full provenance tracking.\n\n"
        "Typical workflow:\n"
        "1. open_project (or init_project for new)\n"
        "2. load_pipeline (from a pipeline.py file)\n"
        "3. source_add_text / source_add_file (feed data)\n"
        "4. build (run transforms, produce snapshot)\n"
        "5. release (materialize search indexes)\n"
        "6. search / get_artifact / lineage (query memory)"
    ),
)

# ---------------------------------------------------------------------------
# Server state — single project, long-lived process
# ---------------------------------------------------------------------------

_state: dict = {"project": None}

_VALID_SEARCH_MODES = {"keyword", "semantic", "hybrid", "layered"}


def _require_project():
    """Return the current project or raise with a clear message."""
    if _state["project"] is None:
        raise ValueError("No project open. Call open_project or init_project first.")
    return _state["project"]


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------


@mcp.tool()
def open_project(path: str = ".") -> dict:
    """Open an existing synix project. Walks upward from path to find .synix/ directory.

    Call this before any other tool. If SYNIX_PROJECT env var is set, the project
    is auto-opened on server start.
    """
    import synix

    project = synix.open_project(path)
    _state["project"] = project
    return {
        "project_root": str(project.project_root),
        "synix_dir": str(project.synix_dir),
        "releases": project.releases(),
    }


@mcp.tool()
def init_project(path: str) -> dict:
    """Create a new synix project at path.

    Creates .synix/ directory structure (objects/, refs/, HEAD).
    Next steps: write a pipeline.py file, then call load_pipeline.
    """
    import synix

    project = synix.init(path)
    _state["project"] = project
    return {
        "project_root": str(project.project_root),
        "synix_dir": str(project.synix_dir),
    }


@mcp.tool()
def load_pipeline(path: str | None = None) -> dict:
    """Load a pipeline definition from a Python file.

    If path is None, looks for pipeline.py in the project root.
    Must be called before build() or source_* operations.

    The pipeline file should define a Pipeline object — see docs/pipeline-api.md.
    """
    project = _require_project()
    pipeline = project.load_pipeline(path)

    from synix.core.models import Source as SourceLayer
    from synix.core.models import Transform

    sources = [layer.name for layer in pipeline.layers if isinstance(layer, SourceLayer)]
    transforms = [layer.name for layer in pipeline.layers if isinstance(layer, Transform)]

    return {
        "name": pipeline.name,
        "source_dir": pipeline.source_dir,
        "sources": sources,
        "transforms": transforms,
        "layer_count": len(pipeline.layers),
    }


# ---------------------------------------------------------------------------
# Build & Release
# ---------------------------------------------------------------------------


@mcp.tool()
def build(
    pipeline_path: str | None = None,
    dry_run: bool = False,
    concurrency: int = 5,
    timeout: float | None = None,
) -> dict:
    """Build the pipeline — run transforms and produce a snapshot.

    Args:
        pipeline_path: Path to pipeline.py (uses loaded pipeline if None).
        dry_run: If True, return plan counts without building.
        concurrency: Max parallel transform workers.
        timeout: Per-request LLM timeout in seconds.

    Returns counts of built, cached, and skipped artifacts.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    project = _require_project()
    if pipeline_path:
        project.load_pipeline(pipeline_path)
    result = project.build(dry_run=dry_run, concurrency=concurrency, timeout=timeout)
    return {
        "built": result.built,
        "cached": result.cached,
        "skipped": result.skipped,
        "total_time": round(result.total_time, 2),
        "snapshot_oid": result.snapshot_oid,
        "manifest_oid": result.manifest_oid,
    }


@mcp.tool()
def release(name: str = "local", ref: str = "HEAD") -> dict:
    """Materialize projections (search indexes, flat files) to a named release.

    Must build() first. Creates .synix/releases/<name>/ with search.db,
    flat files, embeddings, and a receipt.

    Args:
        name: Release name (default "local").
        ref: Snapshot ref to release from (default "HEAD" = latest build).
    """
    project = _require_project()
    return project.release_to(name, ref=ref)


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


@mcp.tool()
def source_list(source_name: str) -> list[str]:
    """List files in a named source directory.

    Source names must match a Source layer declared in the loaded pipeline.
    """
    project = _require_project()
    return project.source(source_name).list()


@mcp.tool()
def source_add_text(source_name: str, content: str, filename: str) -> str:
    """Create a text file in the source directory.

    Use this to add conversations, documents, or any text content
    that the pipeline will process during build.

    Args:
        source_name: Name of the Source layer in the pipeline.
        content: Text content to write.
        filename: Plain filename (no paths — e.g. "chat-01.json").
    """
    project = _require_project()
    project.source(source_name).add_text(content, filename)
    return f"Added {filename} to source {source_name}"


@mcp.tool()
def source_add_file(source_name: str, file_path: str) -> str:
    """Copy an existing file into the source directory.

    Args:
        source_name: Name of the Source layer in the pipeline.
        file_path: Absolute or relative path to the file to copy.
    """
    project = _require_project()
    project.source(source_name).add(file_path)
    return f"Added {Path(file_path).name} to source {source_name}"


@mcp.tool()
def source_remove(source_name: str, filename: str) -> str:
    """Remove a file from the source directory."""
    project = _require_project()
    project.source(source_name).remove(filename)
    return f"Removed {filename} from source {source_name}"


@mcp.tool()
def source_clear(source_name: str) -> str:
    """Remove all files from the source directory."""
    project = _require_project()
    project.source(source_name).clear()
    return f"Cleared all files from source {source_name}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@mcp.tool()
def search(
    query: str,
    release_name: str = "local",
    mode: str = "keyword",
    limit: int = 10,
    layers: list[str] | None = None,
    surface: str | None = None,
) -> list[dict]:
    """Search memory in a named release.

    Args:
        query: Search query string.
        release_name: Release to search (default "local").
        mode: "keyword" (FTS5), "semantic" (embeddings), "hybrid" (both + RRF), or "layered".
        limit: Max results.
        layers: Optional layer name filter.
        surface: Search surface name (required if release has multiple).

    Returns results with content, score, provenance chain, and metadata.
    """
    if mode not in _VALID_SEARCH_MODES:
        raise ValueError(f"Invalid search mode {mode!r}. Must be one of: {sorted(_VALID_SEARCH_MODES)}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    project = _require_project()
    rel = project.release(release_name)
    results = rel.search(query, mode=mode, limit=limit, layers=layers, surface=surface)
    return [
        {
            "label": r.label,
            "layer": r.layer,
            "layer_level": r.layer_level,
            "score": round(r.score, 3),
            "mode": r.mode,
            "content": r.content,
            "provenance": r.provenance,
            "metadata": r.metadata,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


@mcp.tool()
def get_artifact(label: str, release_name: str = "local") -> dict:
    """Read a specific artifact by label.

    Returns full content, metadata, provenance chain, and artifact ID.
    """
    project = _require_project()
    art = project.release(release_name).artifact(label)
    return {
        "label": art.label,
        "artifact_type": art.artifact_type,
        "content": art.content,
        "artifact_id": art.artifact_id,
        "layer": art.layer,
        "layer_level": art.layer_level,
        "provenance": art.provenance,
        "metadata": art.metadata,
    }


@mcp.tool()
def list_artifacts(release_name: str = "local", layer: str | None = None) -> list[dict]:
    """List artifacts in a release (summary only — no content).

    Use get_artifact for full content of a specific artifact.
    """
    project = _require_project()
    arts = list(project.release(release_name).artifacts(layer=layer))
    return [
        {
            "label": a.label,
            "artifact_type": a.artifact_type,
            "layer": a.layer,
            "layer_level": a.layer_level,
            "artifact_id": a.artifact_id,
        }
        for a in arts
    ]


@mcp.tool()
def list_layers(release_name: str = "local") -> list[dict]:
    """List all layers with artifact counts."""
    project = _require_project()
    layers = project.release(release_name).layers()
    return [{"name": l.name, "level": l.level, "count": l.count} for l in layers]


@mcp.tool()
def lineage(label: str, release_name: str = "local") -> list[dict]:
    """Walk the provenance chain for an artifact.

    Returns ancestors in BFS order — from direct parents to root sources.
    Use this to understand how an artifact was derived.
    """
    project = _require_project()
    chain = project.release(release_name).lineage(label)
    return [
        {
            "label": a.label,
            "artifact_type": a.artifact_type,
            "layer": a.layer,
            "layer_level": a.layer_level,
        }
        for a in chain
    ]


@mcp.tool()
def list_releases() -> list[str]:
    """List all named releases."""
    project = _require_project()
    return project.releases()


@mcp.tool()
def show_release(name: str) -> dict:
    """Show release receipt — snapshot OID, adapter results, timing, artifact counts."""
    project = _require_project()
    return project.release(name).receipt()


@mcp.tool()
def get_flat_file(name: str, release_name: str = "local") -> str:
    """Read a flat file projection's rendered markdown content."""
    project = _require_project()
    return project.release(release_name).flat_file(name)


@mcp.tool()
def list_refs() -> dict[str, str]:
    """List all refs — build snapshot OIDs and release pointers."""
    project = _require_project()
    return project.refs()


@mcp.tool()
def clean() -> str:
    """Remove releases/ and work/ directories. Does not remove build snapshots in objects/."""
    project = _require_project()
    project.clean()
    return "Cleaned releases and work directories"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Start the MCP server with stdio transport."""
    project_path = os.environ.get("SYNIX_PROJECT")
    if project_path:
        import synix

        try:
            _state["project"] = synix.open_project(project_path)
            logger.info("Auto-opened project at %s", project_path)
        except Exception:
            logger.error("Failed to auto-open project at %s", project_path, exc_info=True)
            raise

    mcp.run(transport="stdio")
