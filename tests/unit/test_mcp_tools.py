"""Unit tests for Synix MCP server tools — direct function calls, no transport."""

import pytest

import synix
from synix.mcp.server import (
    _require_project,
    _state,
    build,
    clean,
    get_artifact,
    get_flat_file,
    init_project,
    lineage,
    list_artifacts,
    list_layers,
    list_refs,
    list_releases,
    load_pipeline,
    open_project,
    release,
    search,
    show_release,
    source_add_file,
    source_add_text,
    source_clear,
    source_list,
    source_remove,
)

PIPELINE_PY = """\
from synix import Pipeline, Source, SearchSurface, SynixSearch
from synix.transforms import Chunk

pipeline = Pipeline("test-memory", source_dir="./sources")
docs = Source("docs")
chunks = Chunk("chunks", depends_on=[docs], chunk_size=200, chunk_overlap=50)
surface = SearchSurface("search", sources=[chunks], modes=["fulltext"])
search_out = SynixSearch("search", surface=surface)
pipeline.add(docs, chunks, surface, search_out)
"""

DOC_CONTENT = """\
# Chocolate Cake Recipe

The best chocolate cake uses Dutch-process cocoa powder and buttermilk.

Preheat your oven to 350F. Mix dry ingredients: flour, cocoa, sugar,
baking soda, and salt. In a separate bowl, combine buttermilk, eggs,
oil, and vanilla extract. Combine wet and dry, then add hot coffee.

Let cool before frosting with chocolate ganache.
"""


@pytest.fixture(autouse=True)
def reset_mcp_state():
    """Reset MCP server state between tests."""
    _state["project"] = None
    yield
    _state["project"] = None


@pytest.fixture
def project_dir(tmp_path):
    """Create a project with pipeline, sources, and .synix/."""
    (tmp_path / "pipeline.py").write_text(PIPELINE_PY)
    sources = tmp_path / "sources" / "docs"
    sources.mkdir(parents=True)
    (sources / "recipe.md").write_text(DOC_CONTENT)
    synix.init(str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------


class TestProjectLifecycle:
    def test_no_project_raises(self):
        with pytest.raises(ValueError, match="No project open"):
            _require_project()

    def test_init_project(self, tmp_path):
        result = init_project(str(tmp_path / "new-project"))
        assert "project_root" in result
        assert (tmp_path / "new-project" / ".synix").is_dir()

    def test_open_project(self, project_dir):
        result = open_project(str(project_dir))
        assert result["project_root"] == str(project_dir)
        assert "releases" in result

    def test_load_pipeline(self, project_dir):
        open_project(str(project_dir))
        result = load_pipeline()
        assert result["name"] == "test-memory"
        assert "docs" in result["sources"]
        assert "chunks" in result["transforms"]

    def test_load_pipeline_explicit_path(self, project_dir):
        open_project(str(project_dir))
        result = load_pipeline(str(project_dir / "pipeline.py"))
        assert result["name"] == "test-memory"

    def test_load_pipeline_no_project_raises(self):
        with pytest.raises(ValueError, match="No project open"):
            load_pipeline()


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class TestSourceManagement:
    def test_source_list(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        files = source_list("docs")
        assert "recipe.md" in files

    def test_source_add_text(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        result = source_add_text("docs", "Hello world", "hello.txt")
        assert "hello.txt" in result
        assert "hello.txt" in source_list("docs")

    def test_source_add_file(self, project_dir):
        ext_file = project_dir / "external.txt"
        ext_file.write_text("external content")
        open_project(str(project_dir))
        load_pipeline()
        source_add_file("docs", str(ext_file))
        assert "external.txt" in source_list("docs")

    def test_source_remove(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        source_remove("docs", "recipe.md")
        assert "recipe.md" not in source_list("docs")

    def test_source_clear(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        source_clear("docs")
        assert source_list("docs") == []

    def test_source_bad_name_raises(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        with pytest.raises(Exception):
            source_list("nonexistent")


# ---------------------------------------------------------------------------
# Build & Release
# ---------------------------------------------------------------------------


class TestBuildRelease:
    def test_build(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        result = build()
        assert result["built"] > 0
        assert result["snapshot_oid"] is not None

    def test_build_dry_run(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        result = build(dry_run=True)
        assert result["built"] > 0
        assert result["snapshot_oid"] is None

    def test_build_with_pipeline_path(self, project_dir):
        open_project(str(project_dir))
        result = build(pipeline_path=str(project_dir / "pipeline.py"))
        assert result["built"] > 0

    def test_release_after_build(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        build()
        result = release("test-release")
        assert "snapshot_oid" in result

    def test_release_no_build_raises(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        with pytest.raises(Exception):
            release("test-release")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.fixture
    def built_project(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        build()
        release("local")
        return project_dir

    def test_search_keyword(self, built_project):
        results = search("chocolate cake", release_name="local", mode="keyword")
        assert len(results) > 0
        assert any("chocolate" in r["content"].lower() for r in results)

    def test_search_result_structure(self, built_project):
        results = search("cocoa", release_name="local", mode="keyword")
        assert len(results) > 0
        r = results[0]
        for key in ("label", "layer", "layer_level", "score", "mode", "content", "provenance", "metadata"):
            assert key in r, f"Missing key: {key}"

    def test_search_no_results(self, built_project):
        results = search("xyzzynonexistent", release_name="local", mode="keyword")
        assert results == []

    def test_search_limit(self, built_project):
        results = search("chocolate", release_name="local", mode="keyword", limit=1)
        assert len(results) <= 1

    def test_search_layer_filter(self, built_project):
        results = search("chocolate", release_name="local", mode="keyword", layers=["chunks"])
        for r in results:
            assert r["layer"] == "chunks"


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


class TestInspect:
    @pytest.fixture
    def built_project(self, project_dir):
        open_project(str(project_dir))
        load_pipeline()
        build()
        release("local")
        return project_dir

    def test_list_artifacts(self, built_project):
        arts = list_artifacts("local")
        assert len(arts) > 0
        for a in arts:
            for key in ("label", "artifact_type", "layer", "layer_level", "artifact_id"):
                assert key in a

    def test_list_artifacts_filter_layer(self, built_project):
        arts = list_artifacts("local", layer="chunks")
        assert len(arts) > 0
        assert all(a["layer"] == "chunks" for a in arts)

    def test_get_artifact(self, built_project):
        arts = list_artifacts("local")
        label = arts[0]["label"]
        art = get_artifact(label, "local")
        assert art["label"] == label
        assert "content" in art
        assert art["content"]  # not empty

    def test_get_artifact_not_found(self, built_project):
        with pytest.raises(Exception):
            get_artifact("nonexistent-label", "local")

    def test_list_layers(self, built_project):
        layers = list_layers("local")
        assert len(layers) >= 2
        names = [l["name"] for l in layers]
        assert "docs" in names
        assert "chunks" in names
        for l in layers:
            assert "count" in l
            assert l["count"] > 0

    def test_lineage(self, built_project):
        chunk_arts = list_artifacts("local", layer="chunks")
        assert len(chunk_arts) > 0
        chain = lineage(chunk_arts[0]["label"], "local")
        assert len(chain) > 0
        # Should trace back to a source doc
        assert any(a["layer"] == "docs" for a in chain)

    def test_list_releases(self, built_project):
        releases = list_releases()
        assert "local" in releases

    def test_show_release(self, built_project):
        receipt = show_release("local")
        assert "snapshot_oid" in receipt

    def test_show_release_not_found(self, built_project):
        with pytest.raises(Exception):
            show_release("nonexistent")

    def test_list_refs(self, built_project):
        refs = list_refs()
        assert len(refs) > 0

    def test_clean(self, built_project):
        clean()
        assert list_releases() == []
