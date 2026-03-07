"""SynixSearch adapter — builds a self-contained search.db for a release.

The database contains:
- ``search_index`` (FTS5) — full-text searchable artifact content
- ``provenance_chains`` — label -> JSON array of ancestor labels
- ``citation_edges`` — source_label -> target_uri relationships
- ``release_metadata`` — snapshot_oid, manifest_oid, pipeline_name, released_at
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

from synix.build.adapters import AdapterReceipt, ProjectionAdapter, ReleasePlan
from synix.core.citations import extract_citations

logger = logging.getLogger(__name__)


class SynixSearchAdapter(ProjectionAdapter):
    """Materializes a self-contained search.db for a release."""

    def plan(self, closure, declaration, current_receipt=None):
        input_labels = set(declaration.input_artifacts)
        artifacts_to_index = {label: art for label, art in closure.artifacts.items() if label in input_labels}

        if current_receipt and current_receipt.status == "success":
            action = "rebuild"
            summary = f"Rebuild search index with {len(artifacts_to_index)} artifacts"
        else:
            action = "rebuild"
            summary = f"Build search index with {len(artifacts_to_index)} artifacts"

        plan = ReleasePlan(
            adapter="synix_search",
            projection_name=declaration.name,
            action=action,
            summary=summary,
            artifacts_count=len(artifacts_to_index),
        )
        # Stash closure and declaration for apply() — avoids expanding
        # the ReleasePlan signature.
        plan.details["_closure"] = closure
        plan.details["_declaration"] = declaration
        return plan

    def apply(self, plan, target):
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        db_path = target_path / "search.db" if target_path.is_dir() else target_path
        shadow_path = db_path.with_name(f"{db_path.stem}_shadow{db_path.suffix}")

        closure = plan.details.get("_closure")
        declaration = plan.details.get("_declaration")
        if closure is None or declaration is None:
            raise RuntimeError("plan.details must contain '_closure' and '_declaration'")

        input_labels = set(declaration.input_artifacts)
        artifacts = {label: art for label, art in closure.artifacts.items() if label in input_labels}

        try:
            if shadow_path.exists():
                shadow_path.unlink()

            conn = sqlite3.connect(str(shadow_path))
            conn.execute("DROP TABLE IF EXISTS search_index")
            conn.execute(
                """
                CREATE VIRTUAL TABLE search_index USING fts5(
                    content, label, layer_name, layer_level, metadata
                )
                """
            )
            conn.execute("DROP TABLE IF EXISTS citation_edges")
            conn.execute(
                """
                CREATE TABLE citation_edges (
                    source_label TEXT NOT NULL,
                    target_uri TEXT NOT NULL,
                    target_label TEXT,
                    UNIQUE(source_label, target_uri)
                )
                """
            )
            conn.execute("DROP TABLE IF EXISTS provenance_chains")
            conn.execute(
                """
                CREATE TABLE provenance_chains (
                    label TEXT PRIMARY KEY,
                    chain TEXT NOT NULL
                )
                """
            )
            conn.execute("DROP TABLE IF EXISTS release_metadata")
            conn.execute(
                """
                CREATE TABLE release_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.commit()

            # Insert artifacts into the FTS5 index
            for _label, art in artifacts.items():
                metadata = dict(art.metadata)
                # Strip build-internal keys
                for key in ("build_fingerprint", "transform_fingerprint"):
                    metadata.pop(key, None)

                conn.execute(
                    "INSERT INTO search_index "
                    "(content, label, layer_name, layer_level, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        art.content,
                        art.label,
                        art.layer_name,
                        str(art.layer_level),
                        json.dumps(metadata),
                    ),
                )

                # Citation edges
                citations = extract_citations(art.content)
                for cite in citations:
                    conn.execute(
                        "INSERT OR IGNORE INTO citation_edges "
                        "(source_label, target_uri, target_label) "
                        "VALUES (?, ?, ?)",
                        (art.label, cite.uri, cite.ref),
                    )

            # Provenance chains for all artifacts
            for _label, art in artifacts.items():
                conn.execute(
                    "INSERT INTO provenance_chains (label, chain) VALUES (?, ?)",
                    (art.label, json.dumps(art.provenance_chain)),
                )

            # Release metadata
            conn.execute(
                "INSERT INTO release_metadata (key, value) VALUES (?, ?)",
                ("snapshot_oid", closure.snapshot_oid),
            )
            conn.execute(
                "INSERT INTO release_metadata (key, value) VALUES (?, ?)",
                ("manifest_oid", closure.manifest_oid),
            )
            conn.execute(
                "INSERT INTO release_metadata (key, value) VALUES (?, ?)",
                ("pipeline_name", closure.pipeline_name),
            )
            conn.execute(
                "INSERT INTO release_metadata (key, value) VALUES (?, ?)",
                ("released_at", closure.created_at),
            )

            conn.commit()
            conn.close()

            # Atomic swap — replace old db with the new shadow
            db_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(shadow_path), str(db_path))

        except Exception:
            if shadow_path.exists():
                shadow_path.unlink()
            raise

        return AdapterReceipt(
            adapter="synix_search",
            projection_name=plan.projection_name,
            target=str(db_path),
            artifacts_applied=len(artifacts),
            status="success",
        )

    def verify(self, receipt, target):
        target_path = Path(target)
        db_path = target_path / "search.db" if target_path.is_dir() else target_path
        if not db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Check FTS5 table exists
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_index'").fetchone()
            if row is None:
                conn.close()
                return False

            # Check row count matches receipt
            count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
            conn.close()
            return count == receipt.artifacts_applied
        except Exception:
            logger.warning("verify failed for %s", db_path, exc_info=True)
            return False
