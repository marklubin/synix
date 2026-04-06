"""Synix Viewer — web UI for browsing synix releases."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synix.sdk import Project, Release

logger = logging.getLogger(__name__)


def create_app(
    release: Release | None = None,
    *,
    title: str = "Viewer",
    project: Project | None = None,
):
    """Create a Flask app for browsing a synix release.

    Can be started with just a *project* and no release — the viewer
    will show a "waiting for first build" state and discover the release
    once one becomes available.
    """
    from synix.viewer.server import ViewerState
    from synix.viewer.server import create_app as _create_app

    state = ViewerState(release, title, project=project)
    return _create_app(state)


def serve(
    release: Release | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 9471,
    title: str = "Viewer",
    project: Project | None = None,
):
    """Start the viewer dev server.

    If *release* is None but *project* is provided, the viewer starts
    immediately and discovers the release when the first build completes.
    """
    app = create_app(release, title=title, project=project)
    logger.info("Viewer starting on http://%s:%d", host, port)
    app.run(host=host, port=port)
