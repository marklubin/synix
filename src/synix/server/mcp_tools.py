"""Synix knowledge server MCP tools — simplified search/ingest interface.

This is a standalone FastMCP server distinct from synix.mcp.server.
It exposes a minimal tool set for knowledge consumption and ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from synix.workspace import Workspace

logger = logging.getLogger(__name__)

server_mcp = FastMCP(
    "synix-server",
    instructions=(
        "Synix knowledge server — search memory, retrieve context documents, "
        "and ingest new content into configured buckets."
    ),
)

# Global workspace — set by serve.py on startup (replaces old _state dict)
_workspace: Workspace | None = None


_RELEASE_NAME = "local"  # internal — the server always builds and queries this release


def _require_project():
    """Return the current project or raise with a clear message."""
    if _workspace is None:
        raise ValueError("No project open. Server not fully initialized.")
    return _workspace.project


def _current_release():
    """Return the current queryable release, or raise if none exists."""
    project = _require_project()
    try:
        return project.release(_RELEASE_NAME)
    except Exception as exc:
        raise ValueError(
            "No build available yet. Ingest some documents and wait for the auto-builder, or trigger a build manually."
        ) from exc


def _get_runtime_service(attr: str):
    """Get a runtime service by name, or None if not serving."""
    if _workspace is None or _workspace.runtime is None:
        return None
    return getattr(_workspace.runtime, attr, None)


def _require_config():
    """Return the current config or raise with a clear message."""
    if _workspace is None:
        raise ValueError("No config loaded. Server not fully initialized.")
    return _workspace.config


def _resolve_bucket_dir(bucket_name: str) -> Path:
    """Resolve a bucket's directory path, relative to project_dir if not absolute."""
    if _workspace is None:
        raise ValueError("No workspace initialized. Server not fully initialized.")
    return _workspace.bucket_dir(bucket_name)


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
def ingest(bucket: str, content: str, filename: str, client_id: str | None = None) -> str:
    """Write content to a configured ingest bucket and queue for processing.

    Args:
        bucket: Bucket name (must be defined in server config).
        content: Text content to write.
        filename: Filename to create in the bucket directory.
        client_id: Optional client identifier (e.g. "Claude@Salinas").
    """
    import hashlib

    safe_name = _safe_filename(filename)
    bucket_dir = _resolve_bucket_dir(bucket)
    bucket_dir.mkdir(parents=True, exist_ok=True)

    dest = bucket_dir / safe_name
    _atomic_write(dest, content)

    # Enqueue for processing
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    queue = _get_runtime_service("queue")
    doc_id = None
    if queue is not None:
        try:
            doc_id = queue.enqueue(bucket, safe_name, content_hash, str(dest), client_id=client_id)
            logger.info("Ingested %s into bucket %r (doc_id: %s)", safe_name, bucket, doc_id)
        except Exception as exc:
            logger.error("Ingested %s but queue insert failed: %s", safe_name, exc)
    else:
        logger.info("Ingested %s into bucket %r (no queue)", safe_name, bucket)

    result = f"Wrote {safe_name} to bucket {bucket!r} ({dest})"
    if doc_id:
        result += f"\nDocument ID: {doc_id}"
    return result


@server_mcp.tool()
def document_status(doc_id: str) -> str:
    """Check the processing status of an ingested document.

    Args:
        doc_id: Document ID returned by ingest().
    """
    queue = _get_runtime_service("queue")
    if queue is None:
        return "Document queue not initialized."

    status = queue.document_status(doc_id)
    if status is None:
        return f"Document {doc_id} not found."

    lines = [
        f"Document: {doc_id}",
        f"  Status: {status['status']}",
        f"  Bucket: {status['bucket']}",
        f"  File: {status['filename']}",
        f"  Created: {status['created_at']}",
    ]
    if status.get("client_id"):
        lines.append(f"  Client: {status['client_id']}")
    if status.get("queue_position") is not None and status["status"] == "pending":
        lines.append(f"  Queue position: {status['queue_position']}")
    if status.get("processing_started_at"):
        lines.append(f"  Processing started: {status['processing_started_at']}")
    if status.get("built_at"):
        lines.append(f"  Built: {status['built_at']}")
    if status.get("released_at"):
        lines.append(f"  Released: {status['released_at']}")
    if status.get("error_message"):
        lines.append(f"  Error: {status['error_message']}")
    return "\n".join(lines)


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


@server_mcp.tool()
def list_prompts() -> str:
    """List all prompt template keys with version info."""
    store = _get_runtime_service("prompt_store")
    if store is None:
        return "Prompt store not initialized."

    keys = store.list_keys()
    if not keys:
        return "No prompts stored."

    lines = ["Prompt templates:"]
    for key in keys:
        meta = store.get_with_meta(key)
        if meta:
            lines.append(f"  {key} (v{meta['version']}, hash: {meta['content_hash']})")
    return "\n".join(lines)


@server_mcp.tool()
def get_prompt(key: str, version: int | None = None) -> str:
    """Get a prompt template by key.

    Args:
        key: Prompt template key.
        version: Specific version number (latest if omitted).
    """
    store = _get_runtime_service("prompt_store")
    if store is None:
        return "Prompt store not initialized."

    meta = store.get_with_meta(key, version=version)
    if meta is None:
        return f"Prompt '{key}' not found."

    return f"--- {key} v{meta['version']} (hash: {meta['content_hash']}) ---\n{meta['content']}"


@server_mcp.tool()
def update_prompt(key: str, content: str) -> str:
    """Create or update a prompt template.

    Args:
        key: Prompt template key.
        content: New prompt content.
    """
    store = _get_runtime_service("prompt_store")
    if store is None:
        return "Prompt store not initialized."

    result = store.put(key, content)
    return f"Prompt '{key}' updated to v{result['version']} (hash: {result['content_hash']})"


@server_mcp.tool()
def prompt_history(key: str) -> str:
    """Get version history for a prompt template.

    Args:
        key: Prompt template key.
    """
    store = _get_runtime_service("prompt_store")
    if store is None:
        return "Prompt store not initialized."

    hist = store.history(key)
    if not hist:
        return f"No history for prompt '{key}'."

    lines = [f"History for '{key}':"]
    for entry in hist:
        lines.append(f"  v{entry['version']} — {entry['created_at']} (hash: {entry['content_hash']})")
    return "\n".join(lines)
