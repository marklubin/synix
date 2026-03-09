"""E2E test: build -> release -> verify receipt + files."""

from __future__ import annotations

import json
import sqlite3

from click.testing import CliRunner

from synix.build.refs import RefStore
from synix.cli.main import main
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _make_artifact(label, content, atype="episode", layer_name="episodes"):
    return Artifact(
        label=label,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, "layer_level": 0},
    )


def _setup_snapshot(tmp_path, projections=None):
    """Create a snapshot with artifacts and projections."""
    ep1 = _make_artifact("ep-1", "Episode one about testing.")
    ep2 = _make_artifact("ep-2", "Episode two about releases.")
    core = _make_artifact("core-1", "Core memory: user builds agent memory systems.", atype="core", layer_name="core")

    if projections is None:
        projections = {
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-2"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
            "context-doc": {
                "adapter": "flat_file",
                "input_artifacts": ["core-1"],
                "config": {"output_path": "context.md"},
                "config_fingerprint": "sha256:test2",
                "precomputed_oid": None,
            },
        }

    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1, ep2], "core": [core]},
        projections=projections,
    )
    return synix_dir


class TestReleaseCommandE2E:
    def test_release_creates_files(self, tmp_path):
        """synix release HEAD --to local creates search.db + context.md + receipt."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()

        result = runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "Released" in result.output

        release_dir = synix_dir / "releases" / "local"
        assert (release_dir / "receipt.json").exists()
        assert (release_dir / "search.db").exists()
        assert (release_dir / "context.md").exists()

    def test_release_search_db_has_fts5(self, tmp_path):
        """Released search.db has FTS5 table with correct row count."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        db_path = synix_dir / "releases" / "local" / "search.db"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        conn.close()
        assert count == 2  # ep-1 and ep-2

    def test_release_search_db_has_provenance(self, tmp_path):
        """Released search.db has provenance_chains table."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        db_path = synix_dir / "releases" / "local" / "search.db"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM provenance_chains").fetchone()[0]
        conn.close()
        assert count == 2

    def test_release_search_db_has_release_metadata(self, tmp_path):
        """Released search.db has release_metadata table."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        db_path = synix_dir / "releases" / "local" / "search.db"
        conn = sqlite3.connect(str(db_path))
        rows = dict(conn.execute("SELECT key, value FROM release_metadata").fetchall())
        conn.close()
        assert "snapshot_oid" in rows
        assert "pipeline_name" in rows

    def test_release_context_md_content(self, tmp_path):
        """Released context.md contains expected content."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        content = (synix_dir / "releases" / "local" / "context.md").read_text()
        assert "Core memory" in content

    def test_release_json_output(self, tmp_path):
        """synix release --json emits parseable receipt."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()

        result = runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["release_name"] == "local"
        assert "search" in data["adapters"]
        assert "context-doc" in data["adapters"]

    def test_release_advances_ref(self, tmp_path):
        """synix release advances refs/releases/<name>."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        ref_store = RefStore(synix_dir)
        oid = ref_store.read_ref("refs/releases/local")
        assert oid is not None

    def test_release_bad_ref_fails(self, tmp_path):
        """synix release with bad ref exits non-zero."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main, ["release", "refs/heads/nonexistent", "--to", "local", "--synix-dir", str(synix_dir)]
        )
        assert result.exit_code != 0


class TestRevertCommandE2E:
    def test_revert_works(self, tmp_path):
        """synix revert re-releases an older snapshot."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()

        # First release
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        # Revert to the same ref (just testing the command works)
        result = runner.invoke(main, ["revert", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "Reverted" in result.output


class TestReleasesInspectionE2E:
    def test_releases_list(self, tmp_path):
        """synix releases list shows released targets."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        result = runner.invoke(main, ["releases", "list", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "local" in result.output

    def test_releases_list_json(self, tmp_path):
        """synix releases list --json emits machine-readable output."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        result = runner.invoke(main, ["releases", "list", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data["releases"]) == 1

    def test_releases_show(self, tmp_path):
        """synix releases show <name> displays receipt details."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        result = runner.invoke(main, ["releases", "show", "local", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "local" in result.output
        assert "synix_search" in result.output

    def test_releases_show_json(self, tmp_path):
        """synix releases show --json emits receipt dict."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])

        result = runner.invoke(main, ["releases", "show", "local", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["release_name"] == "local"

    def test_releases_show_nonexistent(self, tmp_path):
        """synix releases show <nonexistent> exits non-zero."""
        synix_dir = _setup_snapshot(tmp_path)
        runner = CliRunner()

        result = runner.invoke(main, ["releases", "show", "nope", "--synix-dir", str(synix_dir)])
        assert result.exit_code != 0
