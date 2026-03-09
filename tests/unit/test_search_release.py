"""Unit tests for release-aware search: --release, --ref, ReleaseProvenanceProvider."""

from __future__ import annotations

import json
import sqlite3

import pytest
from click.testing import CliRunner

from synix.cli.main import main
from synix.cli.search_commands import (
    ReleaseProvenanceProvider,
    _list_release_names,
    _resolve_release_db,
)
from synix.core.models import Artifact
from synix.search.indexer import SearchIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_search_db(db_path, artifacts_with_layers: list[tuple[Artifact, str, int]]):
    """Create a search.db with given artifacts."""
    index = SearchIndex(db_path)
    index.create()
    for artifact, layer_name, layer_level in artifacts_with_layers:
        index.insert(artifact, layer_name, layer_level)
    index.close()


def _create_provenance_chains_table(db_path, chains: dict[str, list[str]]):
    """Add a provenance_chains table to an existing search.db."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS provenance_chains (
            label TEXT NOT NULL,
            chain TEXT NOT NULL
        )
    """)
    for label, chain in chains.items():
        conn.execute(
            "INSERT INTO provenance_chains (label, chain) VALUES (?, ?)",
            (label, json.dumps(chain)),
        )
    conn.commit()
    conn.close()


def _setup_release(synix_dir, release_name, artifacts_with_layers, provenance_chains=None):
    """Create a release directory with search.db and optional provenance."""
    release_dir = synix_dir / "releases" / release_name
    release_dir.mkdir(parents=True, exist_ok=True)
    db_path = release_dir / "search.db"
    _create_search_db(db_path, artifacts_with_layers)
    if provenance_chains:
        _create_provenance_chains_table(db_path, provenance_chains)
    return db_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_episode():
    return Artifact(
        label="ep-001",
        artifact_type="episode",
        content="Discussion about machine learning and neural networks",
        metadata={"layer_name": "episodes"},
    )


@pytest.fixture
def sample_rollup():
    return Artifact(
        label="monthly-001",
        artifact_type="rollup",
        content="In March the main themes were machine learning and systems design",
        metadata={"layer_name": "monthly"},
    )


# ---------------------------------------------------------------------------
# ReleaseProvenanceProvider
# ---------------------------------------------------------------------------


class TestReleaseProvenanceProvider:
    def test_reads_chains_from_db(self, tmp_path, sample_episode):
        """Provider reads provenance_chains table from search.db."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])
        _create_provenance_chains_table(
            db_path,
            {
                "ep-001": ["ep-001", "t-001"],
            },
        )

        provider = ReleaseProvenanceProvider(db_path)
        chain = provider.get_chain("ep-001")
        assert len(chain) == 2
        assert chain[0].label == "ep-001"
        assert chain[1].label == "t-001"

    def test_missing_label_returns_self(self, tmp_path, sample_episode):
        """Provider returns [label] for unknown labels."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])
        _create_provenance_chains_table(db_path, {})

        provider = ReleaseProvenanceProvider(db_path)
        chain = provider.get_chain("nonexistent")
        assert len(chain) == 1
        assert chain[0].label == "nonexistent"

    def test_missing_db_returns_empty_provider(self, tmp_path):
        """Provider handles nonexistent db gracefully."""
        provider = ReleaseProvenanceProvider(tmp_path / "does_not_exist.db")
        chain = provider.get_chain("anything")
        assert len(chain) == 1
        assert chain[0].label == "anything"

    def test_no_provenance_table(self, tmp_path, sample_episode):
        """Provider handles db without provenance_chains table."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])
        # No provenance_chains table created

        provider = ReleaseProvenanceProvider(db_path)
        chain = provider.get_chain("ep-001")
        assert len(chain) == 1
        assert chain[0].label == "ep-001"

    def test_get_record_returns_shim(self, tmp_path, sample_episode):
        """get_record returns a shim with parent_labels."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])
        _create_provenance_chains_table(
            db_path,
            {
                "ep-001": ["ep-001", "t-001", "t-002"],
            },
        )

        provider = ReleaseProvenanceProvider(db_path)
        record = provider.get_record("ep-001")
        assert record is not None
        assert record.label == "ep-001"
        assert set(record.parent_labels) == {"t-001", "t-002"}

    def test_get_record_returns_none_for_unknown(self, tmp_path, sample_episode):
        """get_record returns None for unknown labels."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])
        _create_provenance_chains_table(db_path, {})

        provider = ReleaseProvenanceProvider(db_path)
        assert provider.get_record("nonexistent") is None

    def test_malformed_chain_json_logged(self, tmp_path, sample_episode):
        """Malformed JSON in chain column is logged, not swallowed silently."""
        db_path = tmp_path / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE provenance_chains (label TEXT, chain TEXT)
        """)
        conn.execute(
            "INSERT INTO provenance_chains (label, chain) VALUES (?, ?)",
            ("ep-001", "not valid json"),
        )
        conn.commit()
        conn.close()

        # Should not raise, but the chain for ep-001 won't be loaded
        provider = ReleaseProvenanceProvider(db_path)
        chain = provider.get_chain("ep-001")
        # Falls back to [label]
        assert len(chain) == 1
        assert chain[0].label == "ep-001"


# ---------------------------------------------------------------------------
# _list_release_names / _resolve_release_db
# ---------------------------------------------------------------------------


class TestReleaseResolution:
    def test_list_release_names_empty(self, tmp_path):
        """No releases dir returns empty list."""
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir()
        assert _list_release_names(synix_dir) == []

    def test_list_release_names_finds_releases(self, tmp_path, sample_episode):
        """Lists releases that have search.db."""
        synix_dir = tmp_path / ".synix"
        _setup_release(synix_dir, "prod", [(sample_episode, "episodes", 1)])
        _setup_release(synix_dir, "staging", [(sample_episode, "episodes", 1)])

        names = _list_release_names(synix_dir)
        assert names == ["prod", "staging"]

    def test_list_release_names_skips_dirs_without_search_db(self, tmp_path):
        """Directories without search.db are not listed."""
        synix_dir = tmp_path / ".synix"
        releases_dir = synix_dir / "releases" / "empty"
        releases_dir.mkdir(parents=True)

        assert _list_release_names(synix_dir) == []

    def test_resolve_release_db_explicit_name(self, tmp_path, sample_episode):
        """Explicit release name resolves to that release's search.db."""
        synix_dir = tmp_path / ".synix"
        expected_path = _setup_release(synix_dir, "prod", [(sample_episode, "episodes", 1)])

        result = _resolve_release_db(synix_dir, "prod")
        assert result == expected_path

    def test_resolve_release_db_missing_name_exits(self, tmp_path):
        """Explicit release name that doesn't exist raises SystemExit."""
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir(parents=True)

        with pytest.raises(SystemExit):
            _resolve_release_db(synix_dir, "nonexistent")

    def test_resolve_release_db_auto_single(self, tmp_path, sample_episode):
        """Auto-detect works when exactly one release exists."""
        synix_dir = tmp_path / ".synix"
        expected_path = _setup_release(synix_dir, "prod", [(sample_episode, "episodes", 1)])

        result = _resolve_release_db(synix_dir, None)
        assert result == expected_path

    def test_resolve_release_db_auto_multiple_exits(self, tmp_path, sample_episode):
        """Auto-detect with multiple releases raises SystemExit."""
        synix_dir = tmp_path / ".synix"
        _setup_release(synix_dir, "prod", [(sample_episode, "episodes", 1)])
        _setup_release(synix_dir, "staging", [(sample_episode, "episodes", 1)])

        with pytest.raises(SystemExit):
            _resolve_release_db(synix_dir, None)

    def test_resolve_release_db_auto_none(self, tmp_path):
        """Auto-detect with no releases returns None."""
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir(parents=True)

        result = _resolve_release_db(synix_dir, None)
        assert result is None


# ---------------------------------------------------------------------------
# CLI integration: --release flag
# ---------------------------------------------------------------------------


class TestSearchReleaseFlag:
    def test_release_flag_exists(self, runner):
        """CLI accepts --release flag."""
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--release" in result.output

    def test_ref_flag_exists(self, runner):
        """CLI accepts --ref flag."""
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--ref" in result.output

    def test_synix_dir_flag_exists(self, runner):
        """CLI accepts --synix-dir flag."""
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--synix-dir" in result.output

    def test_search_from_release(self, runner, tmp_path, sample_episode, sample_rollup):
        """Search queries a release's search.db when --release is given."""
        synix_dir = tmp_path / ".synix"
        _setup_release(
            synix_dir,
            "prod",
            [
                (sample_episode, "episodes", 1),
                (sample_rollup, "monthly", 2),
            ],
        )

        # Also create an empty build dir so --build-dir doesn't fail
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--release",
                "prod",
                "--synix-dir",
                str(synix_dir),
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output

    def test_search_auto_detects_single_release(self, runner, tmp_path, sample_episode):
        """Search auto-detects a single release when no --release is given."""
        synix_dir = tmp_path / ".synix"
        _setup_release(synix_dir, "prod", [(sample_episode, "episodes", 1)])

        # No build dir search.db — should find the release
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--synix-dir",
                str(synix_dir),
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output

    def test_search_falls_back_to_build_dir(self, runner, tmp_path, sample_episode):
        """Without releases, search falls back to build dir search.db."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        db_path = build_dir / "search.db"
        _create_search_db(db_path, [(sample_episode, "episodes", 1)])

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output

    def test_search_release_with_provenance(self, runner, tmp_path, sample_episode):
        """Release search uses provenance_chains table for --trace."""
        synix_dir = tmp_path / ".synix"
        _setup_release(
            synix_dir,
            "prod",
            [(sample_episode, "episodes", 1)],
            provenance_chains={"ep-001": ["ep-001", "t-001"]},
        )

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--release",
                "prod",
                "--synix-dir",
                str(synix_dir),
                "--build-dir",
                str(build_dir),
                "--trace",
            ],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output
        # Provenance should show t-001
        assert "t-001" in result.output

    def test_search_release_missing_exits(self, runner, tmp_path):
        """Search with --release pointing to nonexistent release exits with error."""
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir(parents=True)

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "test query",
                "--release",
                "nonexistent",
                "--synix-dir",
                str(synix_dir),
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code != 0

    def test_search_release_with_layer_filter(self, runner, tmp_path, sample_episode, sample_rollup):
        """--layers filter works with release-based search."""
        synix_dir = tmp_path / ".synix"
        _setup_release(
            synix_dir,
            "prod",
            [
                (sample_episode, "episodes", 1),
                (sample_rollup, "monthly", 2),
            ],
        )

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--release",
                "prod",
                "--synix-dir",
                str(synix_dir),
                "--build-dir",
                str(build_dir),
                "--layers",
                "episodes",
            ],
        )
        assert result.exit_code == 0
        assert "episodes" in result.output
        # Monthly should not be in results (layer filtered)
        assert "monthly-001" not in result.output

    def test_ref_flag_with_no_snapshot_exits(self, runner, tmp_path):
        """--ref with no snapshot store gives a non-zero exit."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            [
                "search",
                "test",
                "--ref",
                "HEAD",
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_build_dir_flag_still_works(self, runner, tmp_path, sample_episode):
        """--build-dir without --release still works as before."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _create_search_db(build_dir / "search.db", [(sample_episode, "episodes", 1)])

        result = runner.invoke(
            main,
            [
                "search",
                "machine learning",
                "--build-dir",
                str(build_dir),
            ],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output

    def test_projection_flag_still_works(self, runner, tmp_path, sample_episode):
        """--projection resolves custom search output from build dir."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        custom_db = build_dir / "outputs" / "memory.db"
        custom_db.parent.mkdir()
        _create_search_db(custom_db, [(sample_episode, "episodes", 1)])

        (build_dir / ".projection_cache.json").write_text(
            json.dumps(
                {
                    "custom-search": {
                        "projection_type": "synix_search",
                        "db_path": "outputs/memory.db",
                    }
                }
            )
        )

        result = runner.invoke(
            main,
            ["search", "machine learning", "--build-dir", str(build_dir)],
        )
        assert result.exit_code == 0
        assert "ep-001" in result.output

    def test_no_search_index_found(self, runner, tmp_path):
        """When nothing is found, error message is helpful."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = runner.invoke(
            main,
            ["search", "test", "--build-dir", str(build_dir)],
        )
        assert result.exit_code != 0
        assert "No search index found" in result.output
