"""REST API routes for the synix knowledge server."""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from synix.server.mcp_tools import _require_project, _resolve_bucket_dir, _safe_filename

logger = logging.getLogger(__name__)


async def health(request: Request) -> Response:
    """GET /api/v1/health — basic health check."""
    return JSONResponse({"status": "ok"})


async def get_flat_file(request: Request) -> Response:
    """GET /api/v1/flat-file/{name} — return flat file content as markdown."""
    name = request.path_params["name"]
    try:
        project = _require_project()
        rel = project.release("local")
        content = rel.flat_file(name)
        return PlainTextResponse(content, media_type="text/markdown")
    except Exception as exc:
        logger.error("Failed to get flat file %r: %s", name, exc)
        return JSONResponse({"error": str(exc)}, status_code=404)


async def ingest_to_bucket(request: Request) -> Response:
    """POST /api/v1/ingest/{bucket} — write content to a bucket.

    Expects JSON body with "content" and "filename" fields.
    """
    bucket = request.path_params["bucket"]
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    content = body.get("content")
    filename = body.get("filename")

    if not content or not filename:
        return JSONResponse(
            {"error": "Both 'content' and 'filename' fields are required"},
            status_code=400,
        )

    try:
        safe_name = _safe_filename(filename)
        bucket_dir = _resolve_bucket_dir(bucket)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dest = bucket_dir / safe_name
        dest.write_text(content, encoding="utf-8")
        logger.info("REST ingest: %s -> bucket %r", filename, bucket)
        return JSONResponse(
            {"status": "ok", "bucket": bucket, "filename": filename, "path": str(dest)}
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        logger.error("REST ingest failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


api_routes = [
    Route("/api/v1/health", health, methods=["GET"]),
    Route("/api/v1/flat-file/{name:path}", get_flat_file, methods=["GET"]),
    Route("/api/v1/ingest/{bucket}", ingest_to_bucket, methods=["POST"]),
]
