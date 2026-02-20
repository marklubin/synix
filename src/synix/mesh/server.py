"""Mesh HTTP server — Starlette application with all API routes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from synix.mesh.auth import AuthMiddleware
from synix.mesh.config import MeshConfig
from synix.mesh.scheduler import BuildScheduler
from synix.mesh.store import SessionStore

logger = logging.getLogger(__name__)


def create_app(config: MeshConfig) -> Starlette:
    """Create the mesh server Starlette application."""

    mesh_dir = config.mesh_dir
    server_dir = mesh_dir / "server"
    store = SessionStore(
        db_path=server_dir / "sessions.db",
        sessions_dir=server_dir / "sessions",
    )
    scheduler = BuildScheduler(
        min_interval=config.server.build_min_interval,
        batch_threshold=config.server.build_batch_threshold,
        max_delay=config.server.build_max_delay,
    )

    # Track cluster state
    start_time = time.time()
    build_count = 0
    members: dict[str, dict] = {}  # hostname -> {last_heartbeat, term, config_hash}
    current_bundle_etag: str = ""
    current_bundle_path: Path | None = None

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def status(request: Request) -> JSONResponse:
        counts = store.count()
        uptime = time.time() - start_time
        return JSONResponse(
            {
                "status": "ok",
                "build_count": build_count,
                "sessions": counts,
                "uptime_seconds": round(uptime, 1),
                "members": list(members.keys()),
                "scheduler": scheduler.get_status(),
            }
        )

    async def submit_session(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid JSON in session submission: %s", exc)
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        session_id = body.get("session_id")
        project_dir = body.get("project_dir", "default")
        content_b64 = body.get("content")
        expected_sha256 = body.get("sha256")

        if not session_id or not content_b64:
            return JSONResponse(
                {"error": "missing required fields: session_id, content"},
                status_code=400,
            )

        try:
            content = base64.b64decode(content_b64)
        except Exception:
            logger.warning("Invalid base64 content for session %s", session_id, exc_info=True)
            return JSONResponse({"error": "invalid base64 content"}, status_code=400)

        # Validate sha256 if provided
        if expected_sha256:
            actual_sha256 = hashlib.sha256(content).hexdigest()
            if actual_sha256 != expected_sha256:
                return JSONResponse(
                    {"error": "sha256 mismatch", "expected": expected_sha256, "actual": actual_sha256},
                    status_code=400,
                )

        submitted_by = request.headers.get("X-Mesh-Node", "unknown")
        is_new = store.submit(session_id, project_dir, content, submitted_by)

        if is_new:
            await scheduler.notify_new_session()

        return JSONResponse(
            {"session_id": session_id, "new": is_new, "status": "accepted"},
            status_code=201 if is_new else 200,
        )

    async def build_status(request: Request) -> JSONResponse:
        return JSONResponse(scheduler.get_status())

    async def trigger_build(request: Request) -> JSONResponse:
        result = await scheduler.force_rebuild()
        status_code = 200 if result == "started" else 202
        if result == "started":
            asyncio.create_task(_run_build())
        return JSONResponse({"status": result}, status_code=status_code)

    async def artifact_manifest(request: Request) -> JSONResponse:
        build_dir = server_dir / "build"
        manifest_path = build_dir / "manifest.json"
        if not manifest_path.exists():
            return JSONResponse({"error": "no build available"}, status_code=404)

        manifest = json.loads(manifest_path.read_text())
        headers = {}
        if current_bundle_etag:
            headers["ETag"] = current_bundle_etag
        return JSONResponse(manifest, headers=headers)

    async def artifact_bundle(request: Request) -> Response:
        nonlocal current_bundle_path, current_bundle_etag

        if current_bundle_path is None or not current_bundle_path.exists():
            return JSONResponse({"error": "no bundle available"}, status_code=404)

        # Check If-None-Match for conditional request
        if_none_match = request.headers.get("If-None-Match", "")
        if if_none_match and if_none_match == current_bundle_etag:
            return Response(status_code=304)

        bundle_bytes = current_bundle_path.read_bytes()
        return Response(
            content=bundle_bytes,
            media_type="application/gzip",
            headers={
                "ETag": current_bundle_etag,
                "Content-Disposition": f'attachment; filename="{current_bundle_path.name}"',
            },
        )

    async def search(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid JSON in search request: %s", exc)
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        query = body.get("query", "")
        if not query:
            return JSONResponse({"error": "missing required field: query"}, status_code=400)

        search_layers = body.get("layers")
        limit = body.get("limit", 10)
        mode = body.get("mode", "keyword")

        build_dir = server_dir / "build"
        search_db = build_dir / "search.db"
        if not search_db.exists():
            return JSONResponse({"error": "no search index available"}, status_code=404)

        try:
            from synix.build.provenance import ProvenanceTracker
            from synix.search.indexer import SearchIndex
            from synix.search.retriever import HybridRetriever

            index = SearchIndex(search_db)
            provenance = ProvenanceTracker(build_dir)
            retriever = HybridRetriever(
                search_index=index,
                provenance_tracker=provenance,
            )
            results = retriever.query(query, mode=mode, layers=search_layers, top_k=limit)
            return JSONResponse(
                {
                    "results": [
                        {
                            "label": r.label,
                            "layer": r.layer_name,
                            "score": r.score,
                            "content": r.content[:500],
                            "provenance": r.provenance_chain,
                        }
                        for r in results
                    ],
                    "count": len(results),
                }
            )
        except Exception:
            logger.error("Search failed", exc_info=True)
            return JSONResponse({"error": "search failed"}, status_code=500)

    async def heartbeat(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid JSON in heartbeat: %s", exc)
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        hostname = body.get("hostname", "")
        term = body.get("term", 0)
        config_hash_received = body.get("config_hash", "")

        if not hostname:
            return JSONResponse({"error": "missing hostname"}, status_code=400)

        # Config hash validation
        expected_hash = config.cluster.config_hash
        if config_hash_received and config_hash_received != expected_hash:
            return JSONResponse(
                {
                    "error": "config_hash mismatch",
                    "expected": expected_hash,
                    "received": config_hash_received,
                },
                status_code=409,
            )

        # Term fencing — reject stale terms
        current_member = members.get(hostname)
        if current_member and current_member.get("term", 0) > term:
            return JSONResponse(
                {"error": "stale term", "current": current_member["term"], "received": term},
                status_code=409,
            )

        members[hostname] = {
            "last_heartbeat": time.time(),
            "term": term,
            "config_hash": config_hash_received,
        }

        return JSONResponse({"status": "ok", "members": len(members)})

    async def cluster_state(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "members": {
                    hostname: {
                        "last_heartbeat": info["last_heartbeat"],
                        "term": info["term"],
                    }
                    for hostname, info in members.items()
                },
                "config_hash": config.cluster.config_hash,
                "member_count": len(members),
            }
        )

    async def _run_build() -> None:
        nonlocal build_count, current_bundle_etag, current_bundle_path

        try:
            await scheduler.start_build()
        except RuntimeError:
            logger.warning("Build already running, skipping", exc_info=True)
            return

        while True:
            try:
                build_dir = server_dir / "build"
                build_dir.mkdir(parents=True, exist_ok=True)

                # 1. Load pipeline
                from synix.build.pipeline import load_pipeline
                from synix.build.runner import run

                pipeline_path = config.pipeline_path
                pipeline = load_pipeline(pipeline_path)

                # 2. Set source dir to sessions directory
                source_dir = str(server_dir / "sessions")

                # 3. Run build in a thread (sync build system)
                pipeline.build_dir = str(build_dir)
                result = await asyncio.to_thread(run, pipeline, source_dir=source_dir)

                build_count += 1
                logger.info(
                    "Build #%d completed: built=%d cached=%d time=%.1fs",
                    build_count,
                    result.built,
                    result.cached,
                    result.total_time,
                )

                # 4. Run server deploy hooks
                if config.deploy.server_commands:
                    try:
                        from synix.mesh.deploy import run_deploy_hooks

                        await asyncio.to_thread(
                            run_deploy_hooks,
                            config.deploy.server_commands,
                            build_dir,
                        )
                    except Exception:
                        logger.error("Deploy hooks failed", exc_info=True)

                # 5. Create bundle
                try:
                    from synix.mesh.package import create_bundle

                    bundle_path = await asyncio.to_thread(
                        create_bundle,
                        build_dir,
                        config.bundle.include,
                        config.bundle.exclude,
                    )
                    current_bundle_path = bundle_path
                    bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:16]
                    current_bundle_etag = f'"{bundle_hash}"'
                except Exception:
                    logger.error("Bundle creation failed", exc_info=True)

                # 6. Mark sessions as processed
                unprocessed = store.get_unprocessed()
                if unprocessed:
                    session_ids = [s["session_id"] for s in unprocessed]
                    store.mark_processed(session_ids)

                # 7. Send notification
                if config.notifications.webhook_url:
                    try:
                        from synix.mesh.notify import send_notification

                        await send_notification(
                            config.notifications.webhook_url,
                            config.notifications.source,
                            "build_complete",
                            {
                                "build_number": build_count,
                                "built": result.built,
                                "cached": result.cached,
                                "time": result.total_time,
                            },
                        )
                    except Exception:
                        logger.error("Notification failed", exc_info=True)

            except Exception:
                logger.error("Build failed", exc_info=True)

            # 8. Complete scheduler and check if another build is needed
            needs_another = await scheduler.complete_build()
            if not needs_another:
                break

    routes = [
        Route("/api/v1/health", health),
        Route("/api/v1/status", status),
        Route("/api/v1/sessions", submit_session, methods=["POST"]),
        Route("/api/v1/builds/status", build_status),
        Route("/api/v1/builds/trigger", trigger_build, methods=["POST"]),
        Route("/api/v1/artifacts/manifest", artifact_manifest),
        Route("/api/v1/artifacts/bundle", artifact_bundle),
        Route("/api/v1/search", search, methods=["POST"]),
        Route("/api/v1/heartbeat", heartbeat, methods=["POST"]),
        Route("/api/v1/cluster", cluster_state),
    ]

    app = Starlette(routes=routes)
    app.add_middleware(AuthMiddleware, token=config.token)

    # Attach components for testing access
    app.state.store = store
    app.state.scheduler = scheduler

    return app
