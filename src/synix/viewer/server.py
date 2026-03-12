"""Flask app factory for the Synix Viewer, backed by the SDK Release class."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from synix.sdk import ArtifactNotFoundError, Project, Release
from synix.viewer._snippet import make_snippet

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# ViewerState — pre-computed caches built once from the Release
# ---------------------------------------------------------------------------


class ViewerState:
    """Holds the bound release and pre-computed indexes."""

    def __init__(self, release: Release, title: str, *, project: Project | None = None):
        self.release = release
        self.title = title
        self.project = project
        self.metadata_cache: dict[str, dict] = {}
        self.children_index: dict[str, list[str]] = {}
        self._search_cache_key: tuple[str, str | None] = ("", None)
        self._search_cache_results: list = []
        self.ready = False
        self.cache_progress = 0
        self._cache_error: str | None = None

    def start_background_cache(self) -> None:
        """Start building caches in a background thread."""
        thread = threading.Thread(target=self._build_caches, daemon=True)
        thread.start()

    def _build_caches(self) -> None:
        """Iterate every artifact once to populate caches."""
        try:
            start = time.monotonic()
            new_cache: dict[str, dict] = {}
            new_children: dict[str, list[str]] = {}
            count = 0
            for art in self.release.artifacts():
                count += 1
                label = art.label
                meta = art.metadata
                new_cache[label] = {
                    "label": label,
                    "title": meta.get("title", label),
                    "date": meta.get("date") or meta.get("month", ""),
                    "artifact_type": art.artifact_type,
                    "layer": art.layer,
                    "level": art.layer_level,
                    "metadata": meta,
                }

                for parent_label in art.provenance:
                    if parent_label != label:
                        new_children.setdefault(parent_label, []).append(label)
                self.cache_progress = count

            # Atomic swap
            self.metadata_cache = new_cache
            self.children_index = new_children
            self.ready = True
            elapsed = time.monotonic() - start
            logger.info("Viewer ready: %d artifacts cached in %.1fs", count, elapsed)
        except Exception as exc:
            logger.error("Failed to build caches: %s", exc)
            self._cache_error = str(exc)

    def cached_search(self, query: str, layer: str | None = None) -> list:
        """Cache search results for pagination. Clears cache on query/layer change."""
        key = (query, layer)
        if key != self._search_cache_key:
            layers_filter = [layer] if layer else None
            self._search_cache_results = self.release.search(
                query, mode="keyword", limit=500, layers=layers_filter,
            )
            self._search_cache_key = key
        return self._search_cache_results

    def switch_release(self, name: str) -> None:
        """Switch to a different release and rebuild caches atomically.

        Builds new caches before swapping references so concurrent readers
        never see partially-built state.
        """
        if self.project is None:
            raise ValueError("No project available for release switching")
        new_release = self.project.release(name)

        # Build new caches before swapping
        start = time.monotonic()
        new_cache: dict[str, dict] = {}
        new_children: dict[str, list[str]] = {}
        count = 0
        for art in new_release.artifacts():
            count += 1
            label = art.label
            meta = art.metadata
            new_cache[label] = {
                "label": label,
                "title": meta.get("title", label),
                "date": meta.get("date") or meta.get("month", ""),
                "artifact_type": art.artifact_type,
                "layer": art.layer,
                "level": art.layer_level,
                "metadata": meta,
            }
            for parent_label in art.provenance:
                if parent_label != label:
                    new_children.setdefault(parent_label, []).append(label)

        # Atomic swap — readers see either old or new state, never partial
        self.release = new_release
        self.metadata_cache = new_cache
        self.children_index = new_children
        self._search_cache_key = ("", None)
        self._search_cache_results = []
        elapsed = time.monotonic() - start
        logger.info("Release switched to %r: %d artifacts cached in %.1fs", name, count, elapsed)


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

    # -- API ------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "loaded": state.ready,
            "title": state.title,
            "artifact_count": len(state.metadata_cache),
            "release": state.release.name,
            "cache_progress": state.cache_progress,
            "error": state._cache_error,
        })

    @app.route("/api/layers")
    def api_layers():
        layers = state.release.layers()
        result = sorted(
            [{"name": layer.name, "level": layer.level, "count": layer.count} for layer in layers],
            key=lambda l: l["level"],
        )
        return jsonify(result)

    @app.route("/api/artifacts")
    def api_artifacts():
        if not state.ready:
            return jsonify({"error": "Cache is still loading", "cache_progress": state.cache_progress}), 503
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

        items = [m for m in state.metadata_cache.values() if m["layer"] == layer]

        reverse = order == "desc"
        items.sort(key=lambda m: m.get(sort_key, ""), reverse=reverse)

        total = len(items)
        start = (page - 1) * per_page
        page_items = items[start : start + per_page]

        return jsonify({
            "items": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
        })

    @app.route("/api/artifact/<label>")
    def api_artifact(label: str):
        try:
            art = state.release.artifact(label)
        except ArtifactNotFoundError:
            return jsonify({"error": f"Artifact {label!r} not found"}), 404

        return jsonify({
            "label": art.label,
            "artifact_type": art.artifact_type,
            "content": art.content,
            "artifact_id": art.artifact_id,
            "layer": art.layer,
            "layer_level": art.layer_level,
            "provenance": art.provenance,
            "metadata": art.metadata,
        })

    @app.route("/api/lineage/<label>")
    def api_lineage(label: str):
        if not state.ready:
            return jsonify({"error": "Cache is still loading"}), 503
        try:
            parents_raw = state.release.lineage(label)
        except ArtifactNotFoundError:
            return jsonify({"error": f"Artifact {label!r} not found"}), 404

        parents = []
        for art in parents_raw:
            entry = dict(state.metadata_cache.get(art.label, {}))
            if not entry:
                entry = {"label": art.label}
            parents.append(entry)

        children = []
        for child_label in state.children_index.get(label, []):
            entry = dict(state.metadata_cache.get(child_label, {}))
            if not entry:
                entry = {"label": child_label}
            children.append(entry)

        return jsonify({"parents": parents, "children": children})

    @app.route("/api/search")
    def api_search():
        if not state.ready:
            return jsonify({"error": "Cache is still loading"}), 503
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "q parameter is required"}), 400

        layer = request.args.get("layer")
        page = max(1, request.args.get("page", 1, type=int))
        per_page = max(1, min(200, request.args.get("per_page", 20, type=int)))

        all_results = state.cached_search(q, layer)
        total = len(all_results)
        start = (page - 1) * per_page
        page_results = all_results[start : start + per_page]
        has_more = start + per_page < total

        items = []
        for r in page_results:
            items.append({
                "label": r.label,
                "layer": r.layer,
                "score": r.score,
                "snippet": make_snippet(r.content, q),
                "metadata": r.metadata,
            })

        return jsonify({
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "has_more": has_more,
        })

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
        return jsonify({
            "loaded": True,
            "title": state.title,
            "artifact_count": len(state.metadata_cache),
            "release": name,
        })

    return app
