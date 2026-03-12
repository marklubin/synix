"""Flask app factory for the Synix Viewer, backed by the SDK Release class."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from synix.sdk import ArtifactNotFoundError, Release
from synix.viewer._snippet import make_snippet

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# ViewerState — pre-computed caches built once from the Release
# ---------------------------------------------------------------------------


class ViewerState:
    """Holds the bound release and pre-computed indexes."""

    def __init__(self, release: Release, title: str):
        self.release = release
        self.title = title
        self.metadata_cache: dict[str, dict] = {}
        self.children_index: dict[str, list[str]] = {}
        self._build_caches()

    def _build_caches(self) -> None:
        """Iterate every artifact once to populate caches."""
        for art in self.release.artifacts():
            label = art.label
            meta = art.metadata
            self.metadata_cache[label] = {
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
                    self.children_index.setdefault(parent_label, []).append(label)


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
            "artifact_count": len(state.metadata_cache),
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

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 50, type=int)
        sort_key = request.args.get("sort", "date")
        order = request.args.get("order", "desc")

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
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "q parameter is required"}), 400

        layer = request.args.get("layer")
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)

        layers_filter = [layer] if layer else None
        # Fetch one extra result beyond the current page to detect if more exist.
        fetch_limit = page * per_page + 1
        all_results = state.release.search(
            q,
            mode="keyword",
            limit=fetch_limit,
            layers=layers_filter,
        )

        has_more = len(all_results) > page * per_page
        total = len(all_results) if not has_more else len(all_results) - 1
        start = (page - 1) * per_page
        page_results = all_results[start : start + per_page]

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

    return app
