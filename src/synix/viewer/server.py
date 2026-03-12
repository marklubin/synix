"""Flask app factory for the Synix Viewer, backed by the SDK Release class."""

from __future__ import annotations

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

    Nothing is loaded at init.  Layer metadata is fetched from the SDK
    on first access.  The children index (reverse provenance) is built
    lazily on first lineage request.
    """

    def __init__(self, release: Release, title: str, *, project: Project | None = None):
        self.release = release
        self.title = title
        self.project = project
        self._layer_cache: dict[str, list[dict]] = {}
        self._children_index: dict[str, list[str]] | None = None
        self._search_cache_key: tuple[str, str | None] = ("", None)
        self._search_cache_results: list = []

    @property
    def artifact_count(self) -> int:
        """Total artifact count derived from layer metadata (no iteration)."""
        return sum(layer.count for layer in self.release.layers())

    def layer_items(self, layer: str) -> list[dict]:
        """Return metadata dicts for every artifact in *layer*, cached."""
        if layer not in self._layer_cache:
            start = time.monotonic()
            items: list[dict] = []
            for art in self.release.artifacts(layer=layer):
                meta = art.metadata
                items.append({
                    "label": art.label,
                    "title": meta.get("title", art.label),
                    "date": meta.get("date") or meta.get("month", ""),
                    "artifact_type": art.artifact_type,
                    "layer": art.layer,
                    "level": art.layer_level,
                    "metadata": meta,
                })
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
                query, mode="keyword", limit=500, layers=layers_filter,
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

    # -- API ------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "loaded": True,
            "title": state.title,
            "artifact_count": state.artifact_count,
            "release": state.release.name,
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
            "artifact_count": state.artifact_count,
            "release": name,
        })

    return app
