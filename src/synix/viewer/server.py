"""Flask app factory for the Synix Viewer, backed by the SDK Release class."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from synix.sdk import ArtifactNotFoundError, Project, Release, SearchNotAvailableError
from synix.viewer._snippet import make_snippet

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# ViewerState — lazy caches backed by SDK queries
# ---------------------------------------------------------------------------


class ViewerState:
    """Holds the bound release with lazy, per-layer caching.

    Can start with ``release=None`` if only a project is provided.
    In that state, ``has_release`` returns False and data endpoints
    return empty results.  Call ``try_discover_release()`` to check
    if a release has become available (e.g. after the first build).
    """

    def __init__(self, release: Release | None, title: str, *, project: Project | None = None):
        self.release = release
        self.title = title
        self.project = project
        self._workspace = None  # set via from_workspace()
        self._layer_cache: dict[str, list[dict]] = {}
        self._children_index: dict[str, list[str]] | None = None
        self._search_cache_key: tuple[str, str | None] = ("", None)
        self._search_cache_results: list = []
        self._last_discovery_attempt: float = 0.0

    @classmethod
    def from_workspace(cls, workspace, title: str = "Viewer") -> ViewerState:
        """Create ViewerState from a Workspace, discovering release if available."""
        release = None
        try:
            names = workspace.releases()
            if names:
                target = "local" if "local" in names else names[0]
                release = workspace.release(target)
        except Exception:
            logger.debug("Workspace release discovery failed", exc_info=True)
        state = cls(release, title, project=workspace.project)
        state._workspace = workspace
        return state

    @property
    def has_release(self) -> bool:
        return self.release is not None

    def try_discover_release(self) -> bool:
        """Try to discover a release from the project.  Returns True if found.

        Attempts at most once every 5 seconds to avoid per-request filesystem churn.
        If a workspace is bound, uses it for discovery.
        """
        if self.release is not None:
            return True
        if self.project is None and self._workspace is None:
            return False
        # Cooldown: don't re-check more than once per 5 seconds
        now = time.monotonic()
        if now - self._last_discovery_attempt < 5.0:
            return False
        self._last_discovery_attempt = now
        try:
            if self._workspace is not None:
                # Use workspace for discovery
                import synix as _synix

                project = _synix.open_project(str(self._workspace.root))
            else:
                # Re-open to pick up refs created since server start
                import synix as _synix

                project = _synix.open_project(str(self.project.project_root))
            names = project.releases()
            if names:
                target = "local" if "local" in names else names[0]
                self.release = project.release(target)
                self.project = project
                logger.info("Viewer discovered release %r", target)
                return True
        except Exception as exc:
            logger.debug("Release discovery failed: %s", exc)
        return False

    @property
    def artifact_count(self) -> int:
        """Total artifact count derived from layer metadata (no iteration)."""
        if not self.has_release:
            return 0
        return sum(layer.count for layer in self.release.layers())

    def layer_items(self, layer: str) -> list[dict]:
        """Return metadata dicts for every artifact in *layer*, cached."""
        if layer not in self._layer_cache:
            start = time.monotonic()
            items: list[dict] = []
            for art in self.release.artifacts(layer=layer):
                meta = art.metadata
                items.append(
                    {
                        "label": art.label,
                        "title": meta.get("title", art.label),
                        "date": meta.get("date") or meta.get("month", ""),
                        "artifact_type": art.artifact_type,
                        "layer": art.layer,
                        "level": art.layer_level,
                        "metadata": meta,
                    }
                )
            self._layer_cache[layer] = items
            elapsed = time.monotonic() - start
            logger.info("Layer %r: %d artifacts cached in %.1fs", layer, len(items), elapsed)
        return self._layer_cache[layer]

    def children_of(self, label: str) -> list[str]:
        """Return child labels for *label*, building the index lazily."""
        if self._children_index is None:
            start = time.monotonic()
            index: dict[str, list[str]] = {}
            count = 0
            for art in self.release.artifacts():
                count += 1
                for parent_label in art.provenance:
                    if parent_label != art.label:
                        index.setdefault(parent_label, []).append(art.label)
            self._children_index = index
            elapsed = time.monotonic() - start
            logger.info("Children index: %d artifacts scanned in %.1fs", count, elapsed)
        return self._children_index.get(label, [])

    def metadata_for(self, label: str) -> dict | None:
        """Look up cached metadata for a label across all cached layers."""
        for items in self._layer_cache.values():
            for item in items:
                if item["label"] == label:
                    return item
        return None

    def cached_search(self, query: str, layer: str | None = None) -> list:
        """Cache search results for pagination. Clears cache on query/layer change."""
        key = (query, layer)
        if key != self._search_cache_key:
            layers_filter = [layer] if layer else None
            self._search_cache_results = self.release.search(
                query,
                mode="keyword",
                limit=500,
                layers=layers_filter,
            )
            self._search_cache_key = key
        return self._search_cache_results

    def switch_release(self, name: str) -> None:
        """Switch to a different release, clearing all caches."""
        if self.project is None:
            raise ValueError("No project available for release switching")
        self.release = self.project.release(name)
        self._layer_cache.clear()
        self._children_index = None
        self._search_cache_key = ("", None)
        self._search_cache_results = []
        logger.info("Switched to release %r", name)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(state: ViewerState) -> Flask:
    """Create and return a configured Flask application."""

    app = Flask(__name__, static_folder=None)

    # -- Static / SPA ---------------------------------------------------------

    @app.route("/")
    def index():
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.route("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(str(STATIC_DIR), filename)

    # -- Release discovery ----------------------------------------------------

    @app.before_request
    def _try_discover():
        """On each request, try to discover a release if we don't have one yet."""
        if not state.has_release:
            state.try_discover_release()

    def _require_release():
        """Return 503 JSON if no release is available yet."""
        if not state.has_release:
            return jsonify({"error": "No release available yet — waiting for first build"}), 503
        return None

    # -- API ------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        result = {
            "loaded": state.has_release,
            "title": state.title,
            "artifact_count": state.artifact_count,
            "release": state.release.name if state.has_release else None,
        }
        if state._workspace:
            result["workspace"] = state._workspace.name
            result["state"] = state._workspace.state.value
        return jsonify(result)

    @app.route("/api/layers")
    def api_layers():
        err = _require_release()
        if err:
            return err
        layers = state.release.layers()
        result = sorted(
            [{"name": layer.name, "level": layer.level, "count": layer.count} for layer in layers],
            key=lambda l: l["level"],
        )
        return jsonify(result)

    @app.route("/api/artifacts")
    def api_artifacts():
        err = _require_release()
        if err:
            return err
        layer = request.args.get("layer")
        if not layer:
            return jsonify({"error": "layer parameter is required"}), 400

        page = max(1, request.args.get("page", 1, type=int))
        per_page = max(1, min(200, request.args.get("per_page", 50, type=int)))
        sort_key = request.args.get("sort", "date")
        if sort_key not in {"date", "title", "artifact_type"}:
            sort_key = "date"
        order = request.args.get("order", "desc")
        if order not in {"asc", "desc"}:
            order = "desc"

        items = list(state.layer_items(layer))

        reverse = order == "desc"
        items.sort(key=lambda m: m.get(sort_key, ""), reverse=reverse)

        total = len(items)
        start = (page - 1) * per_page
        page_items = items[start : start + per_page]

        return jsonify(
            {
                "items": page_items,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )

    @app.route("/api/artifact/<label>")
    def api_artifact(label: str):
        err = _require_release()
        if err:
            return err
        try:
            art = state.release.artifact(label)
        except ArtifactNotFoundError:
            return jsonify({"error": f"Artifact {label!r} not found"}), 404

        return jsonify(
            {
                "label": art.label,
                "artifact_type": art.artifact_type,
                "content": art.content,
                "artifact_id": art.artifact_id,
                "layer": art.layer,
                "layer_level": art.layer_level,
                "provenance": art.provenance,
                "metadata": art.metadata,
            }
        )

    @app.route("/api/lineage/<label>")
    def api_lineage(label: str):
        err = _require_release()
        if err:
            return err
        try:
            parents_raw = state.release.lineage(label)
        except ArtifactNotFoundError:
            return jsonify({"error": f"Artifact {label!r} not found"}), 404

        parents = []
        for art in parents_raw:
            entry = state.metadata_for(art.label)
            if entry:
                entry = dict(entry)
            else:
                meta = art.metadata
                entry = {
                    "label": art.label,
                    "title": meta.get("title", art.label),
                    "level": art.layer_level,
                }
            parents.append(entry)

        children = []
        for child_label in state.children_of(label):
            entry = state.metadata_for(child_label)
            if entry:
                entry = dict(entry)
            else:
                try:
                    art = state.release.artifact(child_label)
                    meta = art.metadata
                    entry = {
                        "label": art.label,
                        "title": meta.get("title", art.label),
                        "level": art.layer_level,
                    }
                except ArtifactNotFoundError:
                    entry = {"label": child_label}
            children.append(entry)

        return jsonify({"parents": parents, "children": children})

    @app.route("/api/search")
    def api_search():
        err = _require_release()
        if err:
            return err
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "q parameter is required"}), 400

        layer = request.args.get("layer")
        page = max(1, request.args.get("page", 1, type=int))
        per_page = max(1, min(200, request.args.get("per_page", 20, type=int)))

        try:
            all_results = state.cached_search(q, layer)
        except SearchNotAvailableError:
            return jsonify({"error": "Search is not available for this release (no search projection configured)"}), 400
        total = len(all_results)
        start = (page - 1) * per_page
        page_results = all_results[start : start + per_page]
        has_more = start + per_page < total

        items = []
        for r in page_results:
            items.append(
                {
                    "label": r.label,
                    "layer": r.layer,
                    "score": r.score,
                    "snippet": make_snippet(r.content, q),
                    "metadata": r.metadata,
                }
            )

        return jsonify(
            {
                "items": items,
                "total": total,
                "page": page,
                "per_page": per_page,
                "has_more": has_more,
            }
        )

    @app.route("/api/releases")
    def api_releases():
        if state.project is None:
            return jsonify({"releases": [], "current": None})
        names = state.project.releases()
        current = state.release.name
        return jsonify({"releases": names, "current": current})

    @app.route("/api/switch", methods=["POST"])
    def api_switch():
        if state.project is None:
            return jsonify({"error": "No project available"}), 400
        body = request.get_json(silent=True) or {}
        name = body.get("release", "").strip()
        if not name:
            return jsonify({"error": "release parameter required"}), 400
        try:
            state.switch_release(name)
        except Exception as exc:
            logger.error("Failed to switch release to %r: %s", name, exc)
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "loaded": True,
                "title": state.title,
                "artifact_count": state.artifact_count,
                "release": name,
            }
        )

    # -- Helper: prompt store ---------------------------------------------------

    def _get_prompt_store():
        """Get or create a PromptStore for reading."""
        if not hasattr(app, "_prompt_store"):
            if state.project is None:
                return None
            try:
                from synix.server.prompt_store import PromptStore

                prompts_db = Path(state.project._synix_dir) / "prompts.db"
                if prompts_db.exists():
                    app._prompt_store = PromptStore(prompts_db)
                else:
                    return None
            except Exception:
                return None
        return app._prompt_store

    # -- Helper: document queue ------------------------------------------------

    def _get_queue():
        """Get or create a DocumentQueue for reading."""
        if not hasattr(app, "_queue"):
            if state.project is None:
                return None
            try:
                from synix.server.queue import DocumentQueue

                queue_db = Path(state.project._synix_dir) / "queue.db"
                if queue_db.exists():
                    app._queue = DocumentQueue(queue_db)
                else:
                    return None
            except Exception:
                return None
        return app._queue

    # -- Build status, DAG, Prompt management ---------------------------------

    @app.route("/api/build-status")
    def build_status():
        """Queue depth, active build, recent history, stats."""
        if state.project is None:
            return jsonify({"error": "No project"}), 503

        try:
            queue = _get_queue()
            if queue is None:
                return jsonify(
                    {
                        "queue_depth": 0,
                        "active_build": None,
                        "recent": [],
                        "stats": {"total_processed": 0, "avg_build_time_seconds": 0},
                    }
                )

            stats = queue.queue_stats()
            recent = queue.recent_history(limit=20)

            return jsonify(
                {
                    "queue_depth": stats.get("pending_count", 0),
                    "active_build": None,  # TODO: track active build in queue
                    "recent": recent,
                    "stats": stats,
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/dag")
    def dag():
        """Pipeline layer DAG for visualization."""
        if state.project is None or state.project._pipeline is None:
            return jsonify({"error": "No pipeline loaded"}), 503

        try:
            pipeline = state.project._pipeline
            nodes = []
            edges = []

            for layer in pipeline.layers:
                node = {
                    "id": layer.name,
                    "type": type(layer).__name__,
                    "level": getattr(layer, "_level", 0),
                }
                # Try to get artifact count
                try:
                    arts = state.release.artifacts(layer=layer.name)
                    node["count"] = len(list(arts))
                except Exception:
                    node["count"] = 0
                nodes.append(node)

                # Build edges from depends_on
                deps = getattr(layer, "depends_on", None) or []
                for dep in deps:
                    dep_name = dep.name if hasattr(dep, "name") else str(dep)
                    edges.append({"source": dep_name, "target": layer.name})

            # Projections
            projections = []
            for proj in getattr(pipeline, "projections", []):
                proj_data = {
                    "id": proj.name if hasattr(proj, "name") else str(proj),
                    "type": type(proj).__name__,
                }
                sources = getattr(proj, "sources", None) or []
                proj_data["sources"] = [s.name if hasattr(s, "name") else str(s) for s in sources]
                # For SynixSearch, get the surface
                surface = getattr(proj, "surface", None)
                if surface and hasattr(surface, "name"):
                    proj_data["surface"] = surface.name
                projections.append(proj_data)

            return jsonify({"nodes": nodes, "edges": edges, "projections": projections})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # -- Build logs (JSONL) ----------------------------------------------------

    def _get_logs_dir() -> Path | None:
        """Return the logs directory, or None if not available."""
        if state._workspace and hasattr(state._workspace, "synix_dir"):
            return Path(state._workspace.synix_dir) / "logs"
        if state.project and hasattr(state.project, "_synix_dir"):
            return Path(state.project._synix_dir) / "logs"
        return None

    def _parse_build_log(log_path: Path) -> dict:
        """Parse a JSONL build log file into structured JSON."""
        events: list[dict] = []
        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed JSONL line in %s", log_path)
        except OSError as exc:
            logger.warning("Failed to read log file %s: %s", log_path, exc)
            return {"error": f"Failed to read log: {exc}"}

        run_id = log_path.stem
        pipeline_name = ""
        started_at = ""
        completed_at = ""
        total_time = 0.0
        total_llm_calls = 0
        total_tokens = 0

        # Accumulate per-layer data
        layers_data: dict[str, dict] = {}  # name -> layer dict
        layer_order: list[str] = []  # preserve order

        for ev in events:
            event_type = ev.get("event", "")
            ts = ev.get("timestamp", "")

            if event_type == "run_start":
                pipeline_name = ev.get("pipeline", "")
                started_at = ts

            elif event_type == "run_finish":
                completed_at = ts
                total_time = ev.get("total_time", 0.0)
                total_llm_calls = ev.get("total_llm_calls", 0)
                total_tokens = ev.get("total_tokens", 0)

            elif event_type == "layer_start":
                name = ev.get("layer", "")
                if name and name not in layers_data:
                    layers_data[name] = {
                        "name": name,
                        "level": ev.get("level", 0),
                        "built": 0,
                        "cached": 0,
                        "time_seconds": 0.0,
                        "started_at": ts,
                        "finished_at": "",
                        "artifacts": [],
                        "llm_calls": [],
                    }
                    layer_order.append(name)
                elif name in layers_data:
                    layers_data[name]["started_at"] = ts

            elif event_type == "layer_finish":
                name = ev.get("layer", "")
                if name in layers_data:
                    layers_data[name]["built"] = ev.get("built", 0)
                    layers_data[name]["cached"] = ev.get("cached", 0)
                    layers_data[name]["time_seconds"] = ev.get("time_seconds", 0.0)
                    layers_data[name]["finished_at"] = ts

            elif event_type == "artifact_built" or event_type == "artifact_cached":
                name = ev.get("layer", "")
                label = ev.get("label", "")
                if name in layers_data and label:
                    layers_data[name]["artifacts"].append(label)

            elif event_type == "llm_call_finish":
                name = ev.get("layer", "")
                if name in layers_data:
                    layers_data[name]["llm_calls"].append(
                        {
                            "artifact": ev.get("artifact_desc", ""),
                            "duration": ev.get("duration_seconds", 0.0),
                            "input_tokens": ev.get("input_tokens", 0),
                            "output_tokens": ev.get("output_tokens", 0),
                            "model": "",  # model is on llm_call_start, not finish
                        }
                    )

            elif event_type == "llm_call_start":
                # Stash model info — will be paired by the next llm_call_finish
                name = ev.get("layer", "")
                model = ev.get("model", "")
                if name in layers_data and model:
                    # Tag the layer with last-seen model for upcoming finish events
                    layers_data[name].setdefault("_pending_model", model)

        # Backfill model on llm_call entries from llm_call_start events
        # Re-parse start events to pair with finishes
        pending_models: dict[str, list[str]] = {}  # layer -> [model, ...]
        for ev in events:
            event_type = ev.get("event", "")
            name = ev.get("layer", "")
            if event_type == "llm_call_start" and name:
                pending_models.setdefault(name, []).append(ev.get("model", ""))
        for name, layer in layers_data.items():
            models = pending_models.get(name, [])
            for i, call in enumerate(layer.get("llm_calls", [])):
                if i < len(models):
                    call["model"] = models[i]
            # Clean up internal state
            layer.pop("_pending_model", None)

        layers_list = [layers_data[n] for n in layer_order if n in layers_data]

        return {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "total_time": total_time,
            "pipeline": pipeline_name,
            "layers": layers_list,
            "summary": {
                "total_llm_calls": total_llm_calls,
                "total_tokens": total_tokens,
                "layers_count": len(layers_list),
            },
        }

    @app.route("/api/build-logs")
    def build_logs_list():
        """List available JSONL build log files with basic metadata."""
        logs_dir = _get_logs_dir()
        if logs_dir is None or not logs_dir.exists():
            return jsonify({"logs": []})

        log_files = sorted(logs_dir.glob("*.jsonl"), reverse=True)
        logs = []
        for lf in log_files[:50]:  # Cap at 50 most recent
            try:
                size = lf.stat().st_size
            except OSError:
                size = 0
            # Extract timestamp from run_id filename (format: YYYYMMDDTHHMMSSffffffZ-hexhex)
            run_id = lf.stem
            # Parse the timestamp prefix
            ts_str = ""
            try:
                # e.g. 20260406T041136123456Z-abcd1234
                ts_part = run_id.split("-")[0]  # 20260406T041136123456Z
                if ts_part.endswith("Z") and "T" in ts_part:
                    # Reformat to ISO 8601
                    date_part = ts_part[:8]  # 20260406
                    time_part = ts_part[9:-1]  # 041136123456
                    ts_str = (
                        f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T"
                        f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}Z"
                    )
            except (IndexError, ValueError):
                ts_str = run_id

            logs.append(
                {
                    "run_id": run_id,
                    "timestamp": ts_str,
                    "size_bytes": size,
                }
            )

        return jsonify({"logs": logs})

    @app.route("/api/build-log")
    def build_log_detail():
        """Parse and return a specific build log, or the most recent one."""
        logs_dir = _get_logs_dir()
        if logs_dir is None or not logs_dir.exists():
            return jsonify({"error": "No logs directory found"}), 404

        run_id = request.args.get("run_id", "").strip()
        if run_id:
            log_path = logs_dir / f"{run_id}.jsonl"
            if not log_path.exists():
                return jsonify({"error": f"Log file not found: {run_id}"}), 404
        else:
            # Find most recent
            log_files = sorted(logs_dir.glob("*.jsonl"), reverse=True)
            if not log_files:
                return jsonify({"error": "No build logs found"}), 404
            log_path = log_files[0]

        result = _parse_build_log(log_path)
        if "error" in result and len(result) == 1:
            return jsonify(result), 500
        return jsonify(result)

    @app.route("/api/prompts")
    def list_prompts():
        """List all prompt keys with latest version info."""
        try:
            store = _get_prompt_store()
            if store is None:
                return jsonify({"prompts": []})

            keys = store.list_keys()
            prompts = []
            for key in keys:
                meta = store.get_with_meta(key)
                hist = store.history(key)
                prompts.append(
                    {
                        "key": key,
                        "version": meta["version"] if meta else 0,
                        "versions_count": len(hist),
                        "content_hash": meta["content_hash"] if meta else "",
                        "updated_at": meta["created_at"] if meta else "",
                    }
                )
            return jsonify({"prompts": prompts})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/prompts/<key>")
    def get_prompt(key):
        """Get prompt content by key."""
        store = _get_prompt_store()
        if store is None:
            return jsonify({"error": "Prompt store not available"}), 503

        version = request.args.get("version", type=int)
        meta = store.get_with_meta(key, version=version)
        if meta is None:
            return jsonify({"error": f"Prompt '{key}' not found"}), 404
        return jsonify(meta)

    @app.route("/api/prompts/<key>", methods=["PUT"])
    def update_prompt(key):
        """Update prompt content (creates new version)."""
        store = _get_prompt_store()
        if store is None:
            return jsonify({"error": "Prompt store not available"}), 503

        data = request.get_json(silent=True)
        if not data or "content" not in data:
            return jsonify({"error": "'content' field is required"}), 400

        result = store.put(key, data["content"])
        return jsonify(result)

    @app.route("/api/prompts/<key>/history")
    def prompt_history(key):
        """Get version history for a prompt."""
        store = _get_prompt_store()
        if store is None:
            return jsonify({"error": "Prompt store not available"}), 503

        hist = store.history(key)
        return jsonify({"key": key, "versions": hist})

    return app
