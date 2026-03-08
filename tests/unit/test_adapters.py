"""Tests for ProjectionAdapter contract, registry, and built-in adapters."""

from __future__ import annotations

import json
import sqlite3

import pytest

from synix.build.adapters import (
    AdapterReceipt,
    get_adapter,
)
from synix.build.flat_file_adapter import FlatFileAdapter
from synix.build.release import ProjectionDeclaration, ReleaseClosure, ResolvedArtifact
from synix.search.adapter import SynixSearchAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(
    label: str,
    content: str = "Some content",
    layer_name: str = "episodes",
    layer_level: int = 1,
    provenance_chain: list[str] | None = None,
    metadata: dict | None = None,
) -> ResolvedArtifact:
    return ResolvedArtifact(
        label=label,
        artifact_type="episode",
        content=content,
        artifact_id=f"sha256:{label}",
        layer_name=layer_name,
        layer_level=layer_level,
        provenance_chain=provenance_chain or [label],
        metadata=metadata or {"layer_name": layer_name},
    )


def _make_closure(
    artifacts: dict[str, ResolvedArtifact] | None = None,
    projections: dict[str, ProjectionDeclaration] | None = None,
) -> ReleaseClosure:
    if artifacts is None:
        artifacts = {
            "ep-1": _make_artifact("ep-1", content="Episode one content"),
        }
    if projections is None:
        projections = {}
    return ReleaseClosure(
        snapshot_oid="a" * 64,
        manifest_oid="b" * 64,
        pipeline_name="test",
        created_at="2026-03-07T12:00:00Z",
        artifacts=artifacts,
        projections=projections,
    )


def _make_search_declaration(
    name: str = "search",
    input_artifacts: list[str] | None = None,
) -> ProjectionDeclaration:
    return ProjectionDeclaration(
        name=name,
        adapter="synix_search",
        input_artifacts=input_artifacts or ["ep-1"],
        config={"modes": ["fulltext"]},
        config_fingerprint="sha256:cfg1",
    )


def _make_flat_file_declaration(
    name: str = "context-doc",
    input_artifacts: list[str] | None = None,
    config: dict | None = None,
) -> ProjectionDeclaration:
    return ProjectionDeclaration(
        name=name,
        adapter="flat_file",
        input_artifacts=input_artifacts or ["ep-1"],
        config=config or {"output_path": "context.md"},
        config_fingerprint="sha256:cfg2",
    )


# ===========================================================================
# Registry tests
# ===========================================================================


class TestAdapterRegistry:
    def test_get_adapter_synix_search(self):
        adapter = get_adapter("synix_search")
        assert isinstance(adapter, SynixSearchAdapter)

    def test_get_adapter_flat_file(self):
        adapter = get_adapter("flat_file")
        assert isinstance(adapter, FlatFileAdapter)

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter.*'nope'"):
            get_adapter("nope")


# ===========================================================================
# SynixSearchAdapter tests
# ===========================================================================


class TestSynixSearchAdapter:
    def test_search_plan_reports_artifact_count(self):
        """plan() reports the correct count of matching input artifacts."""
        artifacts = {
            "ep-1": _make_artifact("ep-1"),
            "ep-2": _make_artifact("ep-2"),
            "ep-3": _make_artifact("ep-3"),
            "other": _make_artifact("other"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_search_declaration(input_artifacts=["ep-1", "ep-2", "ep-3"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)

        assert plan.artifacts_count == 3
        assert plan.adapter == "synix_search"
        assert plan.action == "rebuild"

    def test_search_apply_creates_db(self, tmp_path):
        """apply() creates a search.db file."""
        closure = _make_closure()
        declaration = _make_search_declaration()

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        db_path = tmp_path / "search.db"
        assert db_path.exists()
        assert receipt.status == "success"
        assert receipt.adapter == "synix_search"

    def test_search_apply_inserts_artifacts(self, tmp_path):
        """apply() inserts the correct number of rows into the FTS5 index."""
        artifacts = {
            "ep-1": _make_artifact("ep-1", content="First episode"),
            "ep-2": _make_artifact("ep-2", content="Second episode"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_search_declaration(input_artifacts=["ep-1", "ep-2"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        assert receipt.artifacts_applied == 2

        conn = sqlite3.connect(str(tmp_path / "search.db"))
        count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        conn.close()
        assert count == 2

    def test_search_apply_creates_provenance_table(self, tmp_path):
        """apply() populates the provenance_chains table."""
        artifacts = {
            "ep-1": _make_artifact("ep-1", provenance_chain=["ep-1", "tx-1"]),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_search_declaration(input_artifacts=["ep-1"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        adapter.apply(plan, tmp_path)

        conn = sqlite3.connect(str(tmp_path / "search.db"))
        row = conn.execute("SELECT chain FROM provenance_chains WHERE label = ?", ("ep-1",)).fetchone()
        conn.close()

        assert row is not None
        chain = json.loads(row[0])
        assert chain == ["ep-1", "tx-1"]

    def test_search_apply_creates_release_metadata(self, tmp_path):
        """apply() populates the release_metadata table."""
        closure = _make_closure()
        declaration = _make_search_declaration()

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        adapter.apply(plan, tmp_path)

        conn = sqlite3.connect(str(tmp_path / "search.db"))
        rows = dict(conn.execute("SELECT key, value FROM release_metadata").fetchall())
        conn.close()

        assert rows["snapshot_oid"] == "a" * 64
        assert rows["manifest_oid"] == "b" * 64
        assert rows["pipeline_name"] == "test"
        # released_at is the actual release time (not snapshot creation time)
        assert "released_at" in rows
        assert rows["released_at"] != ""

    def test_search_apply_creates_citation_edges(self, tmp_path):
        """Artifacts containing synix:// citations produce edge rows."""
        artifacts = {
            "ep-1": _make_artifact(
                "ep-1",
                content="See [episode two](synix://ep-2) for details.",
            ),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_search_declaration(input_artifacts=["ep-1"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        adapter.apply(plan, tmp_path)

        conn = sqlite3.connect(str(tmp_path / "search.db"))
        edges = conn.execute("SELECT source_label, target_uri, target_label FROM citation_edges").fetchall()
        conn.close()

        assert len(edges) == 1
        assert edges[0][0] == "ep-1"
        assert edges[0][1] == "synix://ep-2"
        assert edges[0][2] == "ep-2"

    def test_search_verify_success(self, tmp_path):
        """verify() returns True for a valid db matching the receipt."""
        closure = _make_closure()
        declaration = _make_search_declaration()

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        assert adapter.verify(receipt, tmp_path) is True

    def test_search_verify_missing_db(self, tmp_path):
        """verify() returns False when the db does not exist."""
        receipt = AdapterReceipt(
            adapter="synix_search",
            projection_name="search",
            target=str(tmp_path / "search.db"),
            artifacts_applied=1,
            status="success",
        )

        adapter = SynixSearchAdapter()
        assert adapter.verify(receipt, tmp_path / "nonexistent") is False

    def test_search_verify_wrong_count(self, tmp_path):
        """verify() returns False when row count does not match receipt."""
        closure = _make_closure()
        declaration = _make_search_declaration()

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        # Tamper: claim we applied more artifacts than actually present
        receipt.artifacts_applied = 999

        assert adapter.verify(receipt, tmp_path) is False

    def test_search_surface_sources_filter_correctly(self, tmp_path):
        """input_artifacts selects only the declared subset, not all closure artifacts."""
        artifacts = {
            "ep-1": _make_artifact("ep-1", content="Episode one", layer_name="episodes"),
            "ep-2": _make_artifact("ep-2", content="Episode two", layer_name="episodes"),
            "ep-3": _make_artifact("ep-3", content="Episode three", layer_name="episodes"),
            "rollup-1": _make_artifact("rollup-1", content="Rollup one", layer_name="rollups", layer_level=2),
            "core-1": _make_artifact("core-1", content="Core memory", layer_name="core", layer_level=3),
        }
        closure = _make_closure(artifacts=artifacts)
        # Only surface 3 of 5 artifacts
        declaration = _make_search_declaration(input_artifacts=["ep-1", "rollup-1", "core-1"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        assert plan.artifacts_count == 3

        receipt = adapter.apply(plan, tmp_path)
        assert receipt.artifacts_applied == 3

        conn = sqlite3.connect(str(tmp_path / "search.db"))
        rows = conn.execute("SELECT label FROM search_index ORDER BY label").fetchall()
        conn.close()

        indexed_labels = {row[0] for row in rows}
        assert indexed_labels == {"ep-1", "rollup-1", "core-1"}
        # ep-2 and ep-3 must NOT be indexed
        assert "ep-2" not in indexed_labels
        assert "ep-3" not in indexed_labels

    def test_search_apply_nonempty_release_output(self, tmp_path):
        """After apply(), search.db has non-empty FTS5 index, provenance, and metadata."""
        artifacts = {
            "ep-1": _make_artifact("ep-1", content="First episode content", provenance_chain=["ep-1", "tx-raw"]),
            "ep-2": _make_artifact("ep-2", content="Second episode content", provenance_chain=["ep-2", "tx-raw"]),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_search_declaration(input_artifacts=["ep-1", "ep-2"])

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        assert receipt.status == "success"

        conn = sqlite3.connect(str(tmp_path / "search.db"))

        # FTS5 index is non-empty and contains actual content
        fts_count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        assert fts_count == 2
        contents = conn.execute("SELECT content FROM search_index ORDER BY label").fetchall()
        assert all(row[0] != "" for row in contents)

        # Provenance chains are populated for every indexed artifact
        prov_count = conn.execute("SELECT COUNT(*) FROM provenance_chains").fetchone()[0]
        assert prov_count == 2
        prov_rows = conn.execute("SELECT label, chain FROM provenance_chains ORDER BY label").fetchall()
        for label, chain_json in prov_rows:
            chain = json.loads(chain_json)
            assert len(chain) >= 1, f"provenance chain for {label} should be non-empty"
            assert chain[0] == label, f"provenance chain for {label} should start with itself"

        # Release metadata is populated with all required keys
        meta_rows = dict(conn.execute("SELECT key, value FROM release_metadata").fetchall())
        assert len(meta_rows) >= 4
        for required_key in ("snapshot_oid", "manifest_oid", "pipeline_name", "released_at"):
            assert required_key in meta_rows, f"release_metadata missing key {required_key!r}"
            assert meta_rows[required_key] != "", f"release_metadata[{required_key!r}] should not be empty"

        conn.close()

    def test_search_apply_empty_input_artifacts(self, tmp_path):
        """Declaration with input_artifacts=[] produces a valid but empty index."""
        artifacts = {
            "ep-1": _make_artifact("ep-1", content="Should not be indexed"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = ProjectionDeclaration(
            name="search",
            adapter="synix_search",
            input_artifacts=[],
            config={"modes": ["fulltext"]},
            config_fingerprint="sha256:cfg1",
        )

        adapter = SynixSearchAdapter()
        plan = adapter.plan(closure, declaration)
        assert plan.artifacts_count == 0

        receipt = adapter.apply(plan, tmp_path)
        assert receipt.status == "success"
        assert receipt.artifacts_applied == 0

        db_path = tmp_path / "search.db"
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        fts_count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        assert fts_count == 0

        prov_count = conn.execute("SELECT COUNT(*) FROM provenance_chains").fetchone()[0]
        assert prov_count == 0

        # Release metadata should still be populated even with no artifacts
        meta_rows = dict(conn.execute("SELECT key, value FROM release_metadata").fetchall())
        assert "snapshot_oid" in meta_rows
        assert "pipeline_name" in meta_rows
        conn.close()

    def test_search_apply_atomic_swap(self, tmp_path):
        """If shadow build succeeds, old db is replaced atomically."""
        closure = _make_closure()
        declaration = _make_search_declaration()

        adapter = SynixSearchAdapter()

        # First apply — creates the initial db
        plan1 = adapter.plan(closure, declaration)
        adapter.apply(plan1, tmp_path)
        db_path = tmp_path / "search.db"
        assert db_path.exists()

        # Second apply with different content
        artifacts2 = {
            "ep-1": _make_artifact("ep-1", content="Updated content"),
            "ep-new": _make_artifact("ep-new", content="Brand new"),
        }
        closure2 = _make_closure(artifacts=artifacts2)
        declaration2 = _make_search_declaration(input_artifacts=["ep-1", "ep-new"])
        plan2 = adapter.plan(closure2, declaration2)
        receipt2 = adapter.apply(plan2, tmp_path)

        assert receipt2.artifacts_applied == 2
        assert db_path.exists()

        # Shadow file should not linger
        shadow_path = db_path.with_name("search_shadow.db")
        assert not shadow_path.exists()

        # Verify new content is present
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        conn.close()
        assert count == 2


# ===========================================================================
# FlatFileAdapter tests
# ===========================================================================


class TestFlatFileAdapter:
    def test_flat_file_plan_reports_count(self):
        """plan() reports the correct count of matching input artifacts."""
        artifacts = {
            "core": _make_artifact("core", layer_name="core"),
            "ep-1": _make_artifact("ep-1"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_flat_file_declaration(input_artifacts=["core", "ep-1"])

        adapter = FlatFileAdapter()
        plan = adapter.plan(closure, declaration)

        assert plan.artifacts_count == 2
        assert plan.adapter == "flat_file"
        assert plan.action == "rebuild"

    def test_flat_file_apply_writes_file(self, tmp_path):
        """apply() creates context.md with correct content."""
        artifacts = {
            "core": _make_artifact("core", content="Core memory content"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_flat_file_declaration(input_artifacts=["core"])

        adapter = FlatFileAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        output = tmp_path / "context.md"
        assert output.exists()
        assert output.read_text() == "Core memory content"
        assert receipt.status == "success"
        assert receipt.artifacts_applied == 1

    def test_flat_file_apply_joins_artifacts(self, tmp_path):
        """Multiple artifacts are joined with double newlines."""
        artifacts = {
            "a": _make_artifact("a", content="Part A"),
            "b": _make_artifact("b", content="Part B"),
            "c": _make_artifact("c", content="Part C"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_flat_file_declaration(input_artifacts=["a", "b", "c"])

        adapter = FlatFileAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        output = tmp_path / "context.md"
        assert output.read_text() == "Part A\n\nPart B\n\nPart C"
        assert receipt.artifacts_applied == 3

    def test_flat_file_verify_success(self, tmp_path):
        """verify() returns True when the target file exists."""
        artifacts = {
            "core": _make_artifact("core", content="Content"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_flat_file_declaration(input_artifacts=["core"])

        adapter = FlatFileAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        assert adapter.verify(receipt, tmp_path) is True

    def test_flat_file_verify_missing(self, tmp_path):
        """verify() returns False when the target file does not exist."""
        receipt = AdapterReceipt(
            adapter="flat_file",
            projection_name="context-doc",
            target=str(tmp_path / "nonexistent.md"),
            artifacts_applied=1,
            status="success",
        )

        adapter = FlatFileAdapter()
        assert adapter.verify(receipt, tmp_path) is False

    def test_flat_file_apply_respects_output_path_config(self, tmp_path):
        """Config output_path changes the filename."""
        artifacts = {
            "core": _make_artifact("core", content="Custom output"),
        }
        closure = _make_closure(artifacts=artifacts)
        declaration = _make_flat_file_declaration(
            input_artifacts=["core"],
            config={"output_path": "my-custom-doc.md"},
        )

        adapter = FlatFileAdapter()
        plan = adapter.plan(closure, declaration)
        receipt = adapter.apply(plan, tmp_path)

        output = tmp_path / "my-custom-doc.md"
        assert output.exists()
        assert output.read_text() == "Custom output"
        assert receipt.target == str(output)
