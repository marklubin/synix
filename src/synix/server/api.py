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
    """POST /api/v1/ingest/{bucket} — write content to a bucket and queue for processing."""
    import hashlib

    bucket = request.path_params["bucket"]
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    content = body.get("content")
    filename = body.get("filename")
    client_id = body.get("client_id") or request.headers.get("X-Client-Id")

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

        # Enqueue for processing
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = None
        from synix.server.mcp_tools import _state

        queue = _state.get("queue")
        if queue is not None:
            try:
                doc_id = queue.enqueue(bucket, safe_name, content_hash, str(dest), client_id=client_id)
            except Exception as q_exc:
                logger.error("REST ingest: file written but queue insert failed: %s", q_exc)

        logger.info("REST ingest: %s -> bucket %r (doc_id: %s)", filename, bucket, doc_id)
        response_data = {"status": "ok", "bucket": bucket, "filename": filename, "path": str(dest)}
        if doc_id:
            response_data["doc_id"] = doc_id
        return JSONResponse(response_data)
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
    surface = request.query_params.get("surface", "search")

    logger.info("REST search: q=%r layers=%s surface=%s limit=%d", query, layers_param, surface, limit)

    try:
        rel = _current_release()
    except (ValueError, Exception) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    try:
        layers_list = None
        if layers_param:
            layers_list = [l.strip() for l in layers_param.split(",") if l.strip()]

        results = rel.search(query, mode="keyword", limit=limit, layers=layers_list, surface=surface)
        logger.info("REST search: %d results for %r", len(results), query)

        return JSONResponse(
            {
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
            }
        )
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


async def document_status_api(request: Request) -> Response:
    """GET /api/v1/document/{doc_id} — check document processing status."""
    doc_id = request.path_params["doc_id"]
    from synix.server.mcp_tools import _state

    queue = _state.get("queue")
    if queue is None:
        return JSONResponse({"error": "Document queue not initialized"}, status_code=503)

    status = queue.document_status(doc_id)
    if status is None:
        return JSONResponse({"error": f"Document {doc_id} not found"}, status_code=404)

    return JSONResponse(status)


async def list_prompts_api(request: Request) -> Response:
    """GET /api/v1/prompts — list prompt template keys."""
    from synix.server.mcp_tools import _state

    store = _state.get("prompt_store")
    if store is None:
        return JSONResponse({"prompts": []})

    keys = store.list_keys()
    prompts = []
    for key in keys:
        meta = store.get_with_meta(key)
        prompts.append(meta)
    return JSONResponse({"prompts": prompts})


async def get_prompt_api(request: Request) -> Response:
    """GET /api/v1/prompts/{key} — get prompt content."""
    key = request.path_params["key"]
    from synix.server.mcp_tools import _state

    store = _state.get("prompt_store")
    if store is None:
        return JSONResponse({"error": "Prompt store not available"}, status_code=503)

    version_str = request.query_params.get("version")
    version = int(version_str) if version_str else None

    meta = store.get_with_meta(key, version=version)
    if meta is None:
        return JSONResponse({"error": f"Prompt '{key}' not found"}, status_code=404)
    return JSONResponse(meta)


async def update_prompt_api(request: Request) -> Response:
    """PUT /api/v1/prompts/{key} — update prompt content."""
    key = request.path_params["key"]
    from synix.server.mcp_tools import _state

    store = _state.get("prompt_store")
    if store is None:
        return JSONResponse({"error": "Prompt store not available"}, status_code=503)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    content = body.get("content")
    if not content:
        return JSONResponse({"error": "'content' field is required"}, status_code=400)

    result = store.put(key, content)
    return JSONResponse(result)


async def prompt_history_api(request: Request) -> Response:
    """GET /api/v1/prompts/{key}/history — prompt version history."""
    key = request.path_params["key"]
    from synix.server.mcp_tools import _state

    store = _state.get("prompt_store")
    if store is None:
        return JSONResponse({"error": "Prompt store not available"}, status_code=503)

    hist = store.history(key)
    return JSONResponse({"key": key, "versions": hist})


api_routes = [
    Route("/api/v1/health", health, methods=["GET"]),
    Route("/api/v1/flat-file/{name:path}", get_flat_file, methods=["GET"]),
    Route("/api/v1/ingest/{bucket}", ingest_to_bucket, methods=["POST"]),
    Route("/api/v1/search", search_api, methods=["GET"]),
    Route("/api/v1/buckets", list_buckets_api, methods=["GET"]),
    Route("/api/v1/document/{doc_id}", document_status_api, methods=["GET"]),
    Route("/api/v1/prompts", list_prompts_api, methods=["GET"]),
    Route("/api/v1/prompts/{key:path}/history", prompt_history_api, methods=["GET"]),
    Route("/api/v1/prompts/{key:path}", get_prompt_api, methods=["GET"]),
    Route("/api/v1/prompts/{key:path}", update_prompt_api, methods=["PUT"]),
]
