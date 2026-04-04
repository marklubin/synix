"""Synix knowledge server MCP tools — simplified search/ingest interface.

This is a standalone FastMCP server distinct from synix.mcp.server.
It exposes a minimal tool set for knowledge consumption and ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

server_mcp = FastMCP(
    "synix-server",
    instructions=(
        "Synix knowledge server — search memory, retrieve context documents, "
        "and ingest new content into configured buckets."
    ),
)

# Global state — set by serve.py on startup
_state: dict = {"project": None, "config": None}


_RELEASE_NAME = "local"  # internal — the server always builds and queries this release


def _require_project():
    """Return the current project or raise with a clear message."""
    if _state["project"] is None:
        raise ValueError("No project open. Server not fully initialized.")
    return _state["project"]


def _current_release():
    """Return the current queryable release, or raise if none exists."""
    project = _require_project()
    try:
        return project.release(_RELEASE_NAME)
    except Exception as exc:
        raise ValueError(
            "No build available yet. Ingest some documents and wait for the auto-builder, "
            "or trigger a build manually."
        ) from exc


def _require_config():
    """Return the current config or raise with a clear message."""
    if _state["config"] is None:
        raise ValueError("No config loaded. Server not fully initialized.")
    return _state["config"]


def _resolve_bucket_dir(bucket_name: str) -> Path:
    """Resolve a bucket's directory path, relative to project_dir if not absolute."""
    config = _require_config()
    bucket = None
    for b in config.buckets:
        if b.name == bucket_name:
            bucket = b
            break
    if bucket is None:
        raise ValueError(
            f"Bucket {bucket_name!r} not found. "
            f"Available: {[b.name for b in config.buckets]}"
        )

    bucket_path = Path(bucket.dir)
    if not bucket_path.is_absolute():
        bucket_path = Path(config.project_dir) / bucket_path

    return bucket_path


def _atomic_write(dest: Path, content: str) -> None:
    """Write content to dest via temp file + rename (atomic on same filesystem)."""
    import tempfile

    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(dest)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _safe_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal.

    Strips directory components and rejects empty or dot-only names.
    """
    # Take only the final path component — strips ../ and subdirectories
    safe = Path(filename).name
    if not safe or safe in (".", ".."):
        raise ValueError(f"Invalid filename: {filename!r}")
    return safe


@server_mcp.tool()
def ingest(bucket: str, content: str, filename: str) -> str:
    """Write content to a configured ingest bucket.

    Args:
        bucket: Bucket name (must be defined in server config).
        content: Text content to write.
        filename: Filename to create in the bucket directory.
    """
    safe_name = _safe_filename(filename)
    bucket_dir = _resolve_bucket_dir(bucket)
    bucket_dir.mkdir(parents=True, exist_ok=True)

    dest = bucket_dir / safe_name
    _atomic_write(dest, content)
    logger.info("Ingested %s into bucket %r", safe_name, bucket)
    return f"Wrote {safe_name} to bucket {bucket!r} ({dest})"


@server_mcp.tool()
def search(query: str, layers: str | None = None, limit: int = 10, surface: str = "search") -> str:
    """Search the knowledge base.

    Args:
        query: Search query string.
        layers: Comma-separated layer names to filter (optional).
        limit: Max results (default 10).
        surface: Search surface name (default "search", use "reference" for reference docs).
    """
    logger.info("MCP search: query=%r layers=%s surface=%s limit=%d", query, layers, surface, limit)
    rel = _current_release()

    layers_list = None
    if layers:
        layers_list = [l.strip() for l in layers.split(",") if l.strip()]

    results = rel.search(query, mode="keyword", limit=limit, layers=layers_list, surface=surface)
    logger.info("MCP search: %d results for %r", len(results), query)

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"--- Result {i} (score: {r.score:.3f}, layer: {r.layer}) ---")
        lines.append(r.content)
        lines.append("")

    return "\n".join(lines)


@server_mcp.tool()
def get_context(name: str = "context-doc") -> str:
    """Retrieve a synthesized context document.

    Args:
        name: Projection name (default "context-doc").
    """
    logger.info("MCP get_context: name=%r", name)
    rel = _current_release()
    content = rel.flat_file(name)
    logger.info("MCP get_context: returned %d chars", len(content))
    return content


@server_mcp.tool()
def list_buckets() -> str:
    """List all configured ingest buckets with descriptions."""
    config = _require_config()

    if not config.buckets:
        return "No buckets configured."

    lines = []
    for b in config.buckets:
        desc = f" — {b.description}" if b.description else ""
        patterns = ", ".join(b.patterns)
        lines.append(f"  {b.name}: {b.dir} [{patterns}]{desc}")

    return "Configured buckets:\n" + "\n".join(lines)
