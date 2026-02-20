"""Server source — reads from the mesh session store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from synix.core.models import Artifact, Source

if TYPE_CHECKING:
    from synix.mesh.store import SessionStore

logger = logging.getLogger(__name__)


class GenericServerSource(Source):
    """Source subclass that reads from the mesh session store.

    Instead of reading flat files from a directory, reads gzip-compressed
    session files from the server's SessionStore.
    """

    def __init__(self, name: str = "mesh-source", *, store: SessionStore | None = None):
        super().__init__(name)
        self._store = store

    def load(self, config: dict) -> list[Artifact]:
        """Load unprocessed sessions from store as transcript artifacts.

        Each session becomes one Artifact with:
        - label: f"t-mesh-{session_id}"
        - artifact_type: "transcript"
        - content: decompressed session text
        - metadata: {source: "mesh", session_id, project_dir, submitted_by}
        """
        if self._store is None:
            logger.warning("GenericServerSource has no store configured, returning empty")
            return []

        unprocessed = self._store.get_unprocessed()
        if not unprocessed:
            logger.info("No unprocessed sessions in store")
            return []

        artifacts: list[Artifact] = []
        for session in unprocessed:
            session_id = session["session_id"]
            project_dir = session["project_dir"]
            content_bytes = self._store.get_session_content(session_id, project_dir=project_dir)
            if content_bytes is None:
                logger.warning("Could not read content for session %s, skipping", session_id)
                continue

            content = content_bytes.decode("utf-8", errors="replace")
            artifact = Artifact(
                label=f"t-mesh-{session_id}",
                artifact_type="transcript",
                content=content,
                metadata={
                    "source": "mesh",
                    "session_id": session_id,
                    "project_dir": session["project_dir"],
                    "submitted_by": session["submitted_by"],
                },
            )
            artifacts.append(artifact)

        logger.info("Loaded %d artifacts from mesh session store", len(artifacts))
        return artifacts
