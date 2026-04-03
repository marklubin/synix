"""REST API routes for the synix knowledge server."""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from synix.server.mcp_tools import (
    _atomic_write,
    _current_release,
    _require_config,
    _resolve_bucket_dir,
    _safe_filename,
)

logger = logging.getLogger(__name__)


async def health(request: Request) -> Response:
    """GET /api/v1/health — basic health check."""
    logger.debug("Health check from %s", request.client)
    return JSONResponse({"status": "ok"})


async def get_flat_file(request: Request) -> Response:
    """GET /api/v1/flat-file/{name} — return flat file content as markdown."""
    name = request.path_params["name"]
    logger.info("REST flat-file: name=%r", name)
    try:
        rel = _current_release()
        content = rel.flat_file(name)
        logger.info("REST flat-file: returned %d chars for %r", len(content), name)
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
        _atomic_write(dest, content)
        logger.info("REST ingest: %s -> bucket %r", filename, bucket)
        return JSONResponse(
            {"status": "ok", "bucket": bucket, "filename": filename, "path": str(dest)}
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        logger.error("REST ingest failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def search_api(request: Request) -> Response:
    """GET /api/v1/search?q=...&layers=...&limit=... — search the knowledge base."""
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)

    limit = int(request.query_params.get("limit", "10"))
    layers_param = request.query_params.get("layers")

    logger.info("REST search: q=%r layers=%s limit=%d", query, layers_param, limit)

    try:
        rel = _current_release()
    except (ValueError, Exception) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    try:
        layers_list = None
        if layers_param:
            layers_list = [l.strip() for l in layers_param.split(",") if l.strip()]

        results = rel.search(query, mode="keyword", limit=limit, layers=layers_list)
        logger.info("REST search: %d results for %r", len(results), query)

        return JSONResponse({
            "query": query,
            "count": len(results),
            "results": [
                {
                    "label": r.label,
                    "layer": r.layer,
                    "score": round(r.score, 3),
                    "content": r.content[:500],
                }
                for r in results
            ],
        })
    except Exception as exc:
        logger.error("REST search failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def list_buckets_api(request: Request) -> Response:
    """GET /api/v1/buckets — list configured buckets."""
    try:
        config = _require_config()
        buckets = [
            {
                "name": b.name,
                "dir": b.dir,
                "patterns": b.patterns,
                "description": b.description,
            }
            for b in config.buckets
        ]
        return JSONResponse({"buckets": buckets})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


api_routes = [
    Route("/api/v1/health", health, methods=["GET"]),
    Route("/api/v1/flat-file/{name:path}", get_flat_file, methods=["GET"]),
    Route("/api/v1/ingest/{bucket}", ingest_to_bucket, methods=["POST"]),
    Route("/api/v1/search", search_api, methods=["GET"]),
    Route("/api/v1/buckets", list_buckets_api, methods=["GET"]),
]
