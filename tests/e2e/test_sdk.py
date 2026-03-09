"""E2E tests for the Synix SDK.

All tests use real .synix dirs, real FTS5, real fastembed embeddings, real file I/O.
Only LLM calls are mocked (for build tests).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import synix
from synix.build.release_engine import execute_release
from synix.core.models import Artifact, Pipeline, Source
from synix.sdk import (
    SDK_VERSION,
    ArtifactNotFoundError,
    EmbeddingRequiredError,
    PipelineRequiredError,
    ProjectionNotFoundError,
    Release,
    ReleaseNotFoundError,
    SdkArtifact,
    SdkSearchResult,
    SearchHandle,
    SearchNotAvailableError,
    SynixNotFoundError,
)
from tests.helpers.snapshot_factory import create_test_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk_project(tmp_path: Path) -> Path:
    """Full project: snapshot + release + search.db + embeddings + flat file."""
    ep1 = Artifact(
        label="ep-1",
        artifact_type="episode",
        content="Memory systems and agent architectures.",
        metadata={"layer_level": 1},
    )
    ep2 = Artifact(
        label="ep-2",
        artifact_type="episode",
        content="Release management and deployment pipelines.",
        metadata={"layer_level": 1},
    )
    core = Artifact(
        label="core-1",
        artifact_type="core_memory",
        content="Core memory about software builds.",
        metadata={"layer_level": 2},
    )

    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1, ep2], "core": [core]},
        parent_labels_map={"core-1": ["ep-1", "ep-2"]},
        projections={
            "memory-search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-2", "core-1"],
                "config": {
                    "modes": ["fulltext", "semantic"],
                    "embedding_config": {
                        "provider": "fastembed",
                        "model": "BAAI/bge-small-en-v1.5",
                    },
                },
                "config_fingerprint": "sha256:test",
            },
            "context-doc": {
                "adapter": "flat_file",
                "input_artifacts": ["core-1"],
                "config": {"output_path": "context.md"},
                "config_fingerprint": "sha256:test2",
            },
        },
    )
    execute_release(synix_dir, ref="HEAD", release_name="local")
    return tmp_path


@pytest.fixture
def keyword_only_project(tmp_path: Path) -> Path:
    """Project with keyword-only search (no embedding_config)."""
    ep1 = Artifact(
        label="ep-1",
        artifact_type="episode",
        content="Memory systems and agent architectures.",
    )
    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1]},
        projections={
            "keyword-search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:kw-test",
            },
        },
    )
    execute_release(synix_dir, ref="HEAD", release_name="local")
    return tmp_path


# ---------------------------------------------------------------------------
# SDK Version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_sdk_version(self):
        assert SDK_VERSION == "0.1.0"

    def test_exported_from_package(self):
        assert synix.SDK_VERSION == "0.1.0"


# ---------------------------------------------------------------------------
# Init & Discovery
# ---------------------------------------------------------------------------


class TestInitDiscovery:
    def test_init_creates_synix_dir(self, tmp_path: Path):
        project_dir = tmp_path / "new-project"
        project_dir.mkdir()
        project = synix.init(project_dir)

        assert (project_dir / ".synix").is_dir()
        assert (project_dir / ".synix" / "objects").is_dir()
        assert (project_dir / ".synix" / "refs").is_dir()
        assert (project_dir / ".synix" / "HEAD").exists()
        assert project.project_root == project_dir

    def test_init_with_pipeline_creates_source_dirs(self, tmp_path: Path):
        project_dir = tmp_path / "with-pipeline"
        project_dir.mkdir()

        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("exports"))
        pipeline.add(Source("sessions"))

        project = synix.init(project_dir, pipeline=pipeline)
        assert (project_dir / "sources").is_dir()
        assert (project_dir / "sources" / "exports").is_dir()
        assert (project_dir / "sources" / "sessions").is_dir()

    def test_init_with_custom_source_dir(self, tmp_path: Path):
        project_dir = tmp_path / "custom-src"
        project_dir.mkdir()

        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("exports", dir="./custom/data"))

        project = synix.init(project_dir, pipeline=pipeline)
        assert (project_dir / "custom" / "data").is_dir()

    def test_open_finds_synix_dir(self, sdk_project: Path):
        project = synix.open(sdk_project)
        assert project.synix_dir == sdk_project / ".synix"

    def test_open_walks_upward(self, sdk_project: Path):
        """open() finds .synix in parent directories."""
        subdir = sdk_project / "sub" / "deep"
        subdir.mkdir(parents=True)
        project = synix.open(subdir)
        assert project.synix_dir == sdk_project / ".synix"

    def test_open_raises_not_found(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SynixNotFoundError):
            synix.open(empty)

    def test_init_then_open_roundtrip(self, tmp_path: Path):
        project_dir = tmp_path / "roundtrip"
        project_dir.mkdir()
        synix.init(project_dir)
        project = synix.open(project_dir)
        assert project.synix_dir == project_dir / ".synix"

    def test_open_default_cwd(self, sdk_project: Path, monkeypatch):
        monkeypatch.chdir(sdk_project)
        project = synix.open()
        assert project.synix_dir == sdk_project / ".synix"


# ---------------------------------------------------------------------------
# Pipeline interop
# ---------------------------------------------------------------------------


class TestPipelineInterop:
    def test_set_pipeline(self, sdk_project: Path):
        project = synix.open(sdk_project)
        pipeline = Pipeline("test")
        pipeline.add(Source("data"))
        project.set_pipeline(pipeline)
        assert project.pipeline is pipeline

    def test_load_pipeline_from_file(self, tmp_path: Path):
        """load_pipeline() finds pipeline.py in project root."""
        project_dir = tmp_path / "with-pipeline-file"
        project_dir.mkdir()
        synix.init(project_dir)

        # Create a pipeline.py
        pipeline_py = project_dir / "pipeline.py"
        pipeline_py.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("file-test", source_dir="./sources", build_dir="./build")
            pipeline.add(Source("data"))
            """),
            encoding="utf-8",
        )
        (project_dir / "sources").mkdir()

        project = synix.open(project_dir)
        loaded = project.load_pipeline()
        assert loaded.name == "file-test"
        assert project.pipeline is loaded

    def test_pipeline_required_error(self, tmp_path: Path):
        project_dir = tmp_path / "no-pipeline"
        project_dir.mkdir()
        project = synix.init(project_dir)

        with pytest.raises(PipelineRequiredError):
            project.source("data")

    def test_load_pipeline_explicit_path(self, tmp_path: Path):
        project_dir = tmp_path / "explicit-path"
        project_dir.mkdir()
        synix.init(project_dir)

        pipeline_file = project_dir / "custom.py"
        pipeline_file.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("custom", source_dir="./src", build_dir="./build")
            pipeline.add(Source("raw"))
            """),
            encoding="utf-8",
        )
        (project_dir / "src").mkdir()

        project = synix.open(project_dir)
        loaded = project.load_pipeline(pipeline_file)
        assert loaded.name == "custom"


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class TestSourceManagement:
    def test_add_file(self, tmp_path: Path):
        project_dir = tmp_path / "src-mgmt"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("exports"))
        project = synix.init(project_dir, pipeline=pipeline)

        # Create a source file to add
        data_file = tmp_path / "data.json"
        data_file.write_text('{"hello": "world"}', encoding="utf-8")

        project.source("exports").add(data_file)
        assert "data.json" in project.source("exports").list()

    def test_add_text(self, tmp_path: Path):
        project_dir = tmp_path / "src-text"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("exports"))
        project = synix.init(project_dir, pipeline=pipeline)

        project.source("exports").add_text("hello world", "greeting.txt")
        files = project.source("exports").list()
        assert "greeting.txt" in files

    def test_list_files(self, tmp_path: Path):
        project_dir = tmp_path / "src-list"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("data"))
        project = synix.init(project_dir, pipeline=pipeline)

        project.source("data").add_text("a", "a.txt")
        project.source("data").add_text("b", "b.txt")
        files = project.source("data").list()
        assert files == ["a.txt", "b.txt"]

    def test_remove_file(self, tmp_path: Path):
        project_dir = tmp_path / "src-rm"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("data"))
        project = synix.init(project_dir, pipeline=pipeline)

        project.source("data").add_text("temp", "temp.txt")
        assert "temp.txt" in project.source("data").list()

        project.source("data").remove("temp.txt")
        assert "temp.txt" not in project.source("data").list()

    def test_source_dir_auto_created(self, tmp_path: Path):
        project_dir = tmp_path / "src-auto"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("new-source"))
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("new-source")
        assert (project_dir / "sources" / "new-source").is_dir()
        assert src.list() == []

    def test_custom_source_dir(self, tmp_path: Path):
        project_dir = tmp_path / "src-custom"
        project_dir.mkdir()
        pipeline = Pipeline("test", source_dir="./sources")
        pipeline.add(Source("exports", dir="./custom/exports"))
        project = synix.init(project_dir, pipeline=pipeline)

        project.source("exports").add_text("test", "test.txt")
        assert (project_dir / "custom" / "exports" / "test.txt").exists()


# ---------------------------------------------------------------------------
# Release operations
# ---------------------------------------------------------------------------


class TestRelease:
    def test_release_to_creates_receipt(self, sdk_project: Path):
        """release_to() is tested via fixture — verify receipt format."""
        project = synix.open(sdk_project)
        release = project.release("local")
        receipt = release.receipt()
        assert "release_name" in receipt
        assert receipt["release_name"] == "local"
        assert "snapshot_oid" in receipt
        assert "adapters" in receipt

    def test_release_handle(self, sdk_project: Path):
        project = synix.open(sdk_project)
        release = project.release("local")
        assert release.name == "local"

    def test_releases_list(self, sdk_project: Path):
        project = synix.open(sdk_project)
        names = project.releases()
        assert "local" in names

    def test_release_not_found(self, sdk_project: Path):
        project = synix.open(sdk_project)
        release = project.release("nonexistent")
        with pytest.raises(ReleaseNotFoundError):
            release.receipt()


# ---------------------------------------------------------------------------
# Artifact access
# ---------------------------------------------------------------------------


class TestArtifactAccess:
    def test_artifact_by_label(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        art = release.artifact("ep-1")
        assert isinstance(art, SdkArtifact)
        assert art.label == "ep-1"
        assert art.artifact_type == "episode"
        assert "Memory systems" in art.content

    def test_artifact_content(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        art = release.artifact("core-1")
        assert art.content == "Core memory about software builds."

    def test_artifact_metadata(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        art = release.artifact("ep-1")
        assert isinstance(art.metadata, dict)
        assert art.layer == "episodes"

    def test_artifact_provenance(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        art = release.artifact("core-1")
        assert "core-1" in art.provenance
        # core-1 has parents ep-1, ep-2
        assert "ep-1" in art.provenance
        assert "ep-2" in art.provenance

    def test_artifact_not_found(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        with pytest.raises(ArtifactNotFoundError):
            release.artifact("nonexistent")

    def test_artifacts_all(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        arts = list(release.artifacts())
        labels = {a.label for a in arts}
        assert labels == {"ep-1", "ep-2", "core-1"}

    def test_artifacts_layer_filter(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        episodes = list(release.artifacts(layer="episodes"))
        labels = {a.label for a in episodes}
        assert labels == {"ep-1", "ep-2"}


# ---------------------------------------------------------------------------
# Search — keyword
# ---------------------------------------------------------------------------


class TestSearchKeyword:
    def test_keyword_search(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="keyword")
        assert len(results) > 0
        assert all(isinstance(r, SdkSearchResult) for r in results)
        assert any("memory" in r.content.lower() for r in results)

    def test_keyword_limit(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="keyword", limit=1)
        assert len(results) <= 1

    def test_keyword_layer_filter(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="keyword", layers=["core"])
        for r in results:
            assert r.layer == "core"


# ---------------------------------------------------------------------------
# Search — semantic/hybrid
# ---------------------------------------------------------------------------


class TestSearchSemantic:
    def test_hybrid_search(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("agent memory systems", mode="hybrid")
        assert len(results) > 0
        assert all(r.mode == "hybrid" for r in results)

    def test_semantic_search(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("AI architectures", mode="semantic")
        assert len(results) > 0
        assert all(r.mode == "semantic" for r in results)

    def test_provenance_in_results(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="hybrid")
        # At least some results should have provenance
        for r in results:
            assert isinstance(r.provenance, list)

    def test_layered_search(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("software builds", mode="layered")
        assert len(results) > 0

    def test_hybrid_uses_both(self, sdk_project: Path):
        """Hybrid mode uses both FTS5 and embeddings (has score)."""
        release = synix.open(sdk_project).release("local")
        results = release.search("memory architectures", mode="hybrid")
        assert len(results) > 0
        for r in results:
            assert r.score > 0


# ---------------------------------------------------------------------------
# Search — fail-closed
# ---------------------------------------------------------------------------


class TestSearchFailClosed:
    def test_semantic_with_embeddings_works(self, sdk_project: Path):
        """embedding_config declared + embeddings present → hybrid works."""
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="hybrid")
        assert len(results) > 0

    def test_semantic_missing_embeddings_raises(self, tmp_path: Path):
        """embedding_config declared + embeddings missing → EmbeddingRequiredError."""
        ep1 = Artifact(label="ep-1", artifact_type="episode", content="Test content.")
        synix_dir = create_test_snapshot(
            tmp_path,
            {"episodes": [ep1]},
            projections={
                "memory-search": {
                    "adapter": "synix_search",
                    "input_artifacts": ["ep-1"],
                    "config": {
                        "modes": ["fulltext", "semantic"],
                        "embedding_config": {
                            "provider": "fastembed",
                            "model": "BAAI/bge-small-en-v1.5",
                        },
                    },
                    "config_fingerprint": "sha256:test-missing",
                },
            },
        )
        # Do a release but delete embeddings dir afterward
        execute_release(synix_dir, ref="HEAD", release_name="local")
        import shutil

        shutil.rmtree(synix_dir / "releases" / "local" / "embeddings")

        release = Release(synix_dir, "local")
        with pytest.raises(EmbeddingRequiredError):
            release.search("test", mode="hybrid")

    def test_semantic_no_config_raises(self, keyword_only_project: Path):
        """No embedding_config + mode='semantic' → SearchNotAvailableError."""
        release = synix.open(keyword_only_project).release("local")
        with pytest.raises(SearchNotAvailableError, match="modes="):
            release.search("test", mode="semantic")

    def test_keyword_no_config_works(self, keyword_only_project: Path):
        """No embedding_config + mode='keyword' → works fine."""
        release = synix.open(keyword_only_project).release("local")
        results = release.search("memory", mode="keyword")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Search — surface filtering
# ---------------------------------------------------------------------------


class TestSearchSurface:
    def test_auto_detect_single(self, sdk_project: Path):
        """Default surface auto-detected when only one search projection."""
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="keyword")
        assert len(results) > 0

    def test_explicit_surface(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        results = release.search("memory", mode="keyword", surface="memory-search")
        assert len(results) > 0

    def test_nonexistent_surface(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        with pytest.raises(SearchNotAvailableError):
            release.search("test", surface="nonexistent")

    def test_multiple_surfaces_require_explicit(self, tmp_path: Path):
        """Multiple search projections → must specify surface."""
        ep1 = Artifact(label="ep-1", artifact_type="episode", content="Test content.")
        synix_dir = create_test_snapshot(
            tmp_path,
            {"episodes": [ep1]},
            projections={
                "search-a": {
                    "adapter": "synix_search",
                    "input_artifacts": ["ep-1"],
                    "config": {"modes": ["fulltext"]},
                    "config_fingerprint": "sha256:a",
                },
                "search-b": {
                    "adapter": "synix_search",
                    "input_artifacts": ["ep-1"],
                    "config": {"modes": ["fulltext"]},
                    "config_fingerprint": "sha256:b",
                },
            },
        )
        execute_release(synix_dir, ref="HEAD", release_name="local")

        release = Release(synix_dir, "local")
        with pytest.raises(SearchNotAvailableError, match="Multiple"):
            release.search("test", mode="keyword")


# ---------------------------------------------------------------------------
# SearchHandle
# ---------------------------------------------------------------------------


class TestSearchHandle:
    def test_index_returns_handle(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        handle = release.index("memory-search")
        assert isinstance(handle, SearchHandle)

    def test_handle_search(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        handle = release.index("memory-search")
        results = handle.search("memory", mode="keyword")
        assert len(results) > 0

    def test_handle_fail_closed(self, keyword_only_project: Path):
        release = synix.open(keyword_only_project).release("local")
        handle = release.index("keyword-search")
        with pytest.raises(SearchNotAvailableError):
            handle.search("test", mode="semantic")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class TestLayers:
    def test_layers_list(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        layers = release.layers()
        names = {l.name for l in layers}
        assert "episodes" in names
        assert "core" in names

    def test_layer_counts(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        layers = release.layers()
        layer_map = {l.name: l for l in layers}
        assert layer_map["episodes"].count == 2
        assert layer_map["core"].count == 1

    def test_layer_artifacts(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        layers = release.layers()
        ep_layer = next(l for l in layers if l.name == "episodes")
        arts = list(ep_layer.artifacts())
        assert len(arts) == 2


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_multi_level_provenance(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        chain = release.lineage("core-1")
        labels = [a.label for a in chain]
        assert "core-1" in labels
        assert "ep-1" in labels
        assert "ep-2" in labels

    def test_root_artifact_lineage(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        chain = release.lineage("ep-1")
        # Root artifact provenance is just itself
        assert len(chain) >= 1
        assert chain[0].label == "ep-1"


# ---------------------------------------------------------------------------
# Flat files
# ---------------------------------------------------------------------------


class TestFlatFiles:
    def test_flat_file_content(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        content = release.flat_file("context-doc")
        assert "Core memory about software builds." in content

    def test_flat_file_path(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        path = release.flat_file_path("context-doc")
        assert path.exists()
        assert path.name == "context.md"

    def test_flat_file_not_found(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        with pytest.raises(ProjectionNotFoundError):
            release.flat_file("nonexistent")


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class TestReceipt:
    def test_receipt_schema(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        receipt = release.receipt()
        assert "schema_version" in receipt
        assert "release_name" in receipt
        assert "snapshot_oid" in receipt
        assert "adapters" in receipt

    def test_receipt_has_adapters(self, sdk_project: Path):
        release = synix.open(sdk_project).release("local")
        receipt = release.receipt()
        assert "memory-search" in receipt["adapters"]
        assert "context-doc" in receipt["adapters"]


# ---------------------------------------------------------------------------
# Scratch realization
# ---------------------------------------------------------------------------


class TestScratchRelease:
    def test_head_search(self, sdk_project: Path):
        """release("HEAD").search(...) works via scratch realization."""
        project = synix.open(sdk_project)
        with project.release("HEAD") as release:
            results = release.search("memory", mode="keyword")
            assert len(results) > 0

    def test_context_manager_cleanup(self, sdk_project: Path):
        project = synix.open(sdk_project)
        release = project.release("HEAD")
        with release:
            # Force materialization
            release.search("test", mode="keyword")
            scratch_dir = release._scratch_dir
            assert scratch_dir is not None
            assert scratch_dir.exists()
        # After context manager exit, scratch dir is cleaned up
        assert not scratch_dir.exists()

    def test_explicit_close(self, sdk_project: Path):
        project = synix.open(sdk_project)
        release = project.release("HEAD")
        release.search("test", mode="keyword")
        scratch_dir = release._scratch_dir
        release.close()
        assert not scratch_dir.exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_release_not_found(self, sdk_project: Path):
        project = synix.open(sdk_project)
        release = project.release("does-not-exist")
        with pytest.raises(ReleaseNotFoundError):
            release.artifact("ep-1")

    def test_search_no_db(self, tmp_path: Path):
        """SearchNotAvailableError when no search.db."""
        ep1 = Artifact(label="ep-1", artifact_type="episode", content="Test.")
        synix_dir = create_test_snapshot(
            tmp_path,
            {"episodes": [ep1]},
            projections={
                "context-doc": {
                    "adapter": "flat_file",
                    "input_artifacts": ["ep-1"],
                    "config": {"output_path": "context.md"},
                    "config_fingerprint": "sha256:no-search",
                },
            },
        )
        execute_release(synix_dir, ref="HEAD", release_name="local")

        release = Release(synix_dir, "local")
        with pytest.raises(SearchNotAvailableError, match="No search projections"):
            release.search("test")

    def test_synix_not_found(self, tmp_path: Path):
        with pytest.raises(SynixNotFoundError):
            synix.open(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------


class TestRefs:
    def test_refs_dict(self, sdk_project: Path):
        project = synix.open(sdk_project)
        refs = project.refs()
        assert isinstance(refs, dict)
        # Should have at least heads/main and releases/local
        ref_names = list(refs.keys())
        assert any("heads/main" in r for r in ref_names)
        assert any("releases/local" in r for r in ref_names)


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


class TestClean:
    def test_clean_removes_releases(self, sdk_project: Path):
        project = synix.open(sdk_project)
        releases_dir = project.synix_dir / "releases"
        assert releases_dir.exists()

        project.clean()
        assert not releases_dir.exists()
        # Objects and refs remain
        assert (project.synix_dir / "objects").exists()
        assert (project.synix_dir / "refs").exists()
