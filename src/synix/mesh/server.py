"""Mesh HTTP server — Starlette application with all API routes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from synix.mesh.auth import AuthMiddleware
from synix.mesh.config import MeshConfig
from synix.mesh.logging import mesh_event
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
        quiet_period=config.server.build_quiet_period,
        max_delay=config.server.build_max_delay,
    )

    # Track cluster state — protected by _state_lock (1F)
    start_time = time.time()
    build_count = 0
    members: dict[str, dict] = {}  # hostname -> {last_heartbeat, term_counter, ...}
    current_bundle_etag: str = ""
    current_bundle_path: Path | None = None
    _build_task: asyncio.Task | None = None  # tracked to prevent GC
    _state_lock = asyncio.Lock()

    # Build history ring buffer — last 50 builds
    from collections import deque
    build_history: deque[dict] = deque(maxlen=50)

    # --- Term fencing helper (1C) ---
    async def _check_term(request: Request) -> JSONResponse | None:
        """Check term from request headers. Returns 409 response if stale, None if OK."""
        raw_term = request.headers.get("X-Mesh-Term", "")
        if not raw_term:
            return None  # No term header = no fencing (backward compat)
        try:
            request_term = int(raw_term)
        except ValueError:
            return None
        # Check against highest known term from any member (under lock)
        async with _state_lock:
            max_known_term = 0
            for info in members.values():
                member_term = info.get("term_counter", 0)
                if member_term > max_known_term:
                    max_known_term = member_term
        if request_term < max_known_term:
            return JSONResponse(
                {
                    "error": "stale term",
                    "current_term": max_known_term,
                    "received_term": request_term,
                },
                status_code=409,
            )
        return None

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def status(request: Request) -> JSONResponse:
        counts = store.count()
        uptime = time.time() - start_time
        async with _state_lock:
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
        # Term fencing (1C)
        term_err = await _check_term(request)
        if term_err is not None:
            return term_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid JSON in session submission: %s", exc)
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        session_id = body.get("session_id")
        project_dir = body.get("project_dir", "default")
        subsession_seq = body.get("subsession_seq", 0)
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
        is_new = store.submit(session_id, project_dir, content, submitted_by, subsession_seq=subsession_seq)

        if is_new:
            actual_sha = expected_sha256 or hashlib.sha256(content).hexdigest()
            mesh_event(
                logger,
                logging.INFO,
                f"Session submitted: {session_id} seq={subsession_seq} from {submitted_by}",
                "session_submitted",
                {
                    "hostname": submitted_by,
                    "session_id": session_id,
                    "subsession_seq": subsession_seq,
                    "sha256": actual_sha,
                    "project_dir": project_dir,
                },
            )
            await scheduler.notify_new_session()

        return JSONResponse(
            {"session_id": session_id, "new": is_new, "status": "accepted"},
            status_code=201 if is_new else 200,
        )

    async def build_status(request: Request) -> JSONResponse:
        return JSONResponse(scheduler.get_status())

    async def trigger_build(request: Request) -> JSONResponse:
        nonlocal _build_task

        # Term fencing (1C)
        term_err = await _check_term(request)
        if term_err is not None:
            return term_err

        result = await scheduler.force_rebuild()
        status_code = 200 if result == "started" else 202
        if result == "started":
            async with _state_lock:
                _build_task = asyncio.create_task(_run_build())
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
        """Serve the current build bundle using FileResponse for streaming (1E)."""
        if current_bundle_path is None or not current_bundle_path.exists():
            return JSONResponse({"error": "no bundle available"}, status_code=404)

        # Check If-None-Match for conditional request
        if_none_match = request.headers.get("If-None-Match", "")
        if if_none_match and if_none_match == current_bundle_etag:
            return Response(status_code=304)

        requester = request.headers.get("X-Mesh-Node", "unknown")
        bundle_size = current_bundle_path.stat().st_size
        mesh_event(
            logger,
            logging.INFO,
            f"Artifact pull by {requester}",
            "artifact_pull",
            {
                "hostname": requester,
                "etag": current_bundle_etag,
                "bundle_size": bundle_size,
            },
        )
        return FileResponse(
            path=str(current_bundle_path),
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
            from synix.build.refs import synix_dir_for_build_dir
            from synix.build.snapshot_view import SnapshotArtifactCache
            from synix.search.indexer import SearchIndex
            from synix.search.retriever import HybridRetriever

            index = SearchIndex(search_db)
            synix_dir = synix_dir_for_build_dir(build_dir)
            provenance = SnapshotArtifactCache(synix_dir)
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
        raw_term = body.get("term", 0)
        config_hash_received = body.get("config_hash", "")

        if not hostname:
            return JSONResponse({"error": "missing hostname"}, status_code=400)

        # Normalize term — accept dict {"counter": N, "leader_id": S} or bare int
        if isinstance(raw_term, dict):
            term_counter = raw_term.get("counter", 0)
            term_leader = raw_term.get("leader_id", "")
        else:
            term_counter = int(raw_term)
            term_leader = ""

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

        async with _state_lock:
            # Term fencing — reject stale terms (compare counter as int)
            current_member = members.get(hostname)
            if current_member:
                stored_counter = current_member.get("term_counter", 0)
                if stored_counter > term_counter:
                    return JSONResponse(
                        {
                            "error": "stale term",
                            "current_counter": stored_counter,
                            "received_counter": term_counter,
                        },
                        status_code=409,
                    )

            members[hostname] = {
                "last_heartbeat": time.time(),
                "term_counter": term_counter,
                "term_leader": term_leader,
                "config_hash": config_hash_received,
            }

        mesh_event(
            logger,
            logging.DEBUG,
            f"Heartbeat from {hostname}",
            "heartbeat_received",
            {
                "hostname": hostname,
                "term_counter": term_counter,
                "config_hash": config_hash_received,
            },
        )
        return JSONResponse({"status": "ok", "members": len(members)})

    async def cluster_info(request: Request) -> JSONResponse:
        async with _state_lock:
            return JSONResponse(
                {
                    "members": {
                        hostname: {
                            "last_heartbeat": info["last_heartbeat"],
                            "term": {
                                "counter": info.get("term_counter", 0),
                                "leader_id": info.get("term_leader", ""),
                            },
                        }
                        for hostname, info in members.items()
                    },
                    "config_hash": config.cluster.config_hash,
                    "member_count": len(members),
                }
            )

    # --- Session sync endpoints (Part 2) ---
    _sessions_manifest_etag: str = ""

    async def sessions_manifest(request: Request) -> Response:
        """Return JSON manifest of all sessions for client sync."""
        nonlocal _sessions_manifest_etag

        all_sessions = store.list_all_sessions()
        # Compute ETag from sorted session IDs + project_dirs + subsession_seq
        etag_payload = "|".join(
            f"{s['session_id']}:{s['project_dir']}:{s.get('subsession_seq', 0)}" for s in all_sessions
        )
        etag = f'"{hashlib.sha256(etag_payload.encode()).hexdigest()[:16]}"'
        _sessions_manifest_etag = etag

        # Check If-None-Match
        if_none_match = request.headers.get("If-None-Match", "")
        if if_none_match and if_none_match == etag:
            return Response(status_code=304)

        return JSONResponse(
            {
                "sessions": [
                    {
                        "session_id": s["session_id"],
                        "project_dir": s["project_dir"],
                        "subsession_seq": s.get("subsession_seq", 0),
                        "jsonl_sha256": s["jsonl_sha256"],
                    }
                    for s in all_sessions
                ],
                "count": len(all_sessions),
            },
            headers={"ETag": etag},
        )

    async def session_file(request: Request) -> Response:
        """Download a raw .jsonl.gz session file."""
        session_id = request.path_params["session_id"]
        project_dir = request.query_params.get("project_dir", "default")
        subsession_seq = int(request.query_params.get("subsession_seq", "0"))

        file_path = store.get_session_file_path(session_id, project_dir, subsession_seq=subsession_seq)
        if file_path is None:
            return JSONResponse(
                {"error": "session file not found", "session_id": session_id, "project_dir": project_dir},
                status_code=404,
            )

        return FileResponse(
            path=str(file_path),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{file_path.name}"',
            },
        )

    async def builds_history(request: Request) -> JSONResponse:
        """Return build history (last 50 builds, newest first)."""
        return JSONResponse({"builds": list(reversed(build_history))})

    async def logs_tail(request: Request) -> JSONResponse:
        """Return recent structured log entries from the JSON log file."""
        limit = min(int(request.query_params.get("limit", "100")), 500)
        level_filter = request.query_params.get("level", "").upper()
        event_filter = request.query_params.get("event", "")

        log_path = mesh_dir / "logs" / "server.log"
        if not log_path.exists():
            return JSONResponse({"entries": [], "log_file": str(log_path)})

        # Read last N lines efficiently (read from end)
        entries: list[dict] = []
        try:
            raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in reversed(raw_lines):
                if len(entries) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if level_filter and entry.get("level", "") != level_filter:
                    continue
                if event_filter and entry.get("event", "") != event_filter:
                    continue
                entries.append(entry)
        except OSError as exc:
            return JSONResponse({"error": f"Cannot read log: {exc}"}, status_code=500)

        return JSONResponse({"entries": entries, "count": len(entries)})

    async def _run_build() -> None:
        nonlocal build_count, current_bundle_etag, current_bundle_path

        try:
            await scheduler.start_build()
        except RuntimeError:
            logger.warning("Build already running, skipping", exc_info=True)
            return

        pending_count = scheduler.pending_count
        mesh_event(
            logger,
            logging.INFO,
            f"Build started ({pending_count} pending)",
            "build_started",
            {
                "pending_count": pending_count,
            },
        )

        while True:
            build_start = time.time()
            # Snapshot unprocessed sessions BEFORE the build so we only mark
            # these as processed afterward — sessions arriving mid-build stay pending.
            pre_build_unprocessed = store.get_unprocessed()
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

                async with _state_lock:
                    build_count += 1
                    local_build_count = build_count

                build_duration = time.time() - build_start
                mesh_event(
                    logger,
                    logging.INFO,
                    f"Build #{local_build_count} completed: built={result.built} cached={result.cached}",
                    "build_completed",
                    {
                        "build_number": local_build_count,
                        "duration": round(build_duration, 1),
                        "built": result.built,
                        "cached": result.cached,
                    },
                )
                build_history.append({
                    "build_number": local_build_count,
                    "status": "success",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(build_start)),
                    "duration": round(build_duration, 1),
                    "built": result.built,
                    "cached": result.cached,
                    "sessions_processed": len(pre_build_unprocessed),
                })

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

                # 5. Create bundle — pre-compute ETag (1E)
                try:
                    from synix.mesh.package import create_bundle

                    bundle_path = await asyncio.to_thread(
                        create_bundle,
                        build_dir,
                        config.bundle.include,
                        config.bundle.exclude,
                    )
                    # Stream hash computation — avoid reading full bundle into memory
                    h = hashlib.sha256()
                    bundle_size = 0
                    with open(bundle_path, "rb") as f:
                        while True:
                            chunk = f.read(65536)
                            if not chunk:
                                break
                            h.update(chunk)
                            bundle_size += len(chunk)
                    bundle_hash = h.hexdigest()[:16]
                    async with _state_lock:
                        current_bundle_path = bundle_path
                        current_bundle_etag = f'"{bundle_hash}"'
                    mesh_event(
                        logger,
                        logging.INFO,
                        f"Bundle created: {bundle_path.name}",
                        "bundle_created",
                        {
                            "size_bytes": bundle_size,
                            "etag": f'"{bundle_hash}"',
                        },
                    )
                except Exception:
                    logger.error("Bundle creation failed", exc_info=True)

                # 6. Post-build callback (e.g. release to viewer)
                if config.post_build_callback is not None:
                    try:
                        cb = config.post_build_callback
                        if asyncio.iscoroutinefunction(cb):
                            await cb(build_dir, local_build_count)
                        else:
                            await asyncio.to_thread(cb, build_dir, local_build_count)
                    except Exception:
                        logger.error("Post-build callback failed", exc_info=True)

                # 7. Mark only the pre-build snapshot as processed (not mid-build arrivals)
                if pre_build_unprocessed:
                    session_keys = [
                        (s["session_id"], s["project_dir"], s.get("subsession_seq", 0))
                        for s in pre_build_unprocessed
                    ]
                    store.mark_processed(session_keys)

                # 8. Send notification
                if config.notifications.webhook_url:
                    try:
                        from synix.mesh.notify import send_notification

                        await send_notification(
                            config.notifications.webhook_url,
                            config.notifications.source,
                            "build_complete",
                            {
                                "build_number": local_build_count,
                                "built": result.built,
                                "cached": result.cached,
                                "time": result.total_time,
                            },
                        )
                    except Exception:
                        logger.error("Notification failed", exc_info=True)

            except Exception as exc:
                mesh_event(logger, logging.ERROR, f"Build failed: {exc}", "build_failed", {"error": str(exc)})
                build_history.append({
                    "build_number": build_count + 1,
                    "status": "failed",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(build_start)),
                    "duration": round(time.time() - build_start, 1),
                    "error": str(exc),
                    "sessions_processed": len(pre_build_unprocessed),
                })

            # 9. Complete scheduler and check if another build is needed
            needs_another = await scheduler.complete_build()
            if not needs_another:
                break

    # --- Scheduler background loop (1B) ---
    async def _scheduler_loop() -> None:
        """Poll scheduler and auto-start builds when thresholds are met."""
        nonlocal _build_task
        while True:
            await asyncio.sleep(30)
            try:
                if await scheduler.should_build():
                    async with _state_lock:
                        if _build_task is None or _build_task.done():
                            _build_task = asyncio.create_task(_run_build())
            except Exception:
                logger.warning("Scheduler loop check failed", exc_info=True)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        """Start background scheduler loop on startup."""
        task = asyncio.create_task(_scheduler_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    routes = [
        Route("/api/v1/health", health),
        Route("/api/v1/status", status),
        Route("/api/v1/sessions", submit_session, methods=["POST"]),
        Route("/api/v1/sessions/manifest", sessions_manifest),
        Route("/api/v1/sessions/{session_id}/file", session_file),
        Route("/api/v1/builds/status", build_status),
        Route("/api/v1/builds/history", builds_history),
        Route("/api/v1/builds/trigger", trigger_build, methods=["POST"]),
        Route("/api/v1/logs", logs_tail),
        Route("/api/v1/artifacts/manifest", artifact_manifest),
        Route("/api/v1/artifacts/bundle", artifact_bundle),
        Route("/api/v1/search", search, methods=["POST"]),
        Route("/api/v1/heartbeat", heartbeat, methods=["POST"]),
        Route("/api/v1/cluster", cluster_info),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(AuthMiddleware, token=config.token)

    # Attach components for testing access
    app.state.store = store
    app.state.scheduler = scheduler

    return app
