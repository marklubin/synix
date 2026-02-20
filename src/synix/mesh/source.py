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

        Each session/subsession becomes one Artifact with:
        - label: f"t-mesh-{session_id}" (seq=0) or f"t-mesh-{session_id}-sub{seq:04d}" (seq>0)
        - artifact_type: "transcript"
        - content: decompressed session text
        - metadata: {source, session_id, subsession_seq, parent_session_id, project_dir, submitted_by}
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
            seq = session.get("subsession_seq", 0)

            content_bytes = self._store.get_session_content(
                session_id,
                project_dir=project_dir,
                subsession_seq=seq,
            )
            if content_bytes is None:
                logger.warning(
                    "Could not read content for session %s seq=%d, skipping",
                    session_id,
                    seq,
                )
                continue

            content = content_bytes.decode("utf-8", errors="replace")

            # Label: backward compat for seq=0, subsession suffix for seq>0
            if seq > 0:
                label = f"t-mesh-{session_id}-sub{seq:04d}"
            else:
                label = f"t-mesh-{session_id}"

            artifact = Artifact(
                label=label,
                artifact_type="transcript",
                content=content,
                metadata={
                    "source": "mesh",
                    "session_id": session_id,
                    "subsession_seq": seq,
                    "parent_session_id": session_id,
                    "project_dir": project_dir,
                    "submitted_by": session["submitted_by"],
                },
            )
            artifacts.append(artifact)

        logger.info("Loaded %d artifacts from mesh session store", len(artifacts))
        return artifacts
