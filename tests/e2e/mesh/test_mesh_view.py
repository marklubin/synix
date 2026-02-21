"""E2E: Interactive memory viewer — render functions and dispatch."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.core.models import Artifact
from synix.mesh.viewer import (
    MeshContext,
    ViewState,
    dispatch,
    render_artifacts,
    render_builds,
    render_config,
    render_detail,
    render_overview,
    render_pipeline,
    render_search,
)


@pytest.fixture
def build_dir(tmp_path):
    """Create a build dir with some test artifacts."""
    bd = tmp_path / "build"
    bd.mkdir()
    store = ArtifactStore(bd)

    # Transcripts (L0)
    for i in range(3):
        store.save_artifact(
            Artifact(
                label=f"tx-conv-{i}",
                artifact_type="transcript",
                content=f"# Conversation {i}\n\nHello world from conversation {i}.",
                metadata={"date": f"2025-01-0{i + 1}", "title": f"Conv {i}"},
            ),
            layer_name="transcripts",
            layer_level=0,
        )

    # Episodes (L1)
    for i in range(2):
        store.save_artifact(
            Artifact(
                label=f"ep-conv-{i}",
                artifact_type="episode",
                content=f"# Episode {i}\n\nSummary of conversation {i}.",
                metadata={"date": f"2025-01-0{i + 1}", "title": f"Episode {i}"},
            ),
            layer_name="episodes",
            layer_level=1,
        )

    # Core (L3)
    store.save_artifact(
        Artifact(
            label="core-v1",
            artifact_type="core_memory",
            content="# Core Memory\n\nThe user prefers Python.",
            metadata={"title": "Core Memory v1"},
        ),
        layer_name="core",
        layer_level=3,
    )

    # Provenance
    prov = ProvenanceTracker(bd)
    prov.record("ep-conv-0", parent_labels=["tx-conv-0"])
    prov.record("ep-conv-1", parent_labels=["tx-conv-1"])
    prov.record("core-v1", parent_labels=["ep-conv-0", "ep-conv-1"])

    return bd


@pytest.fixture
def mesh_ctx(tmp_path, build_dir):
    """Create a MeshContext for testing (no live server)."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()

    # Write a minimal config
    config_content = f"""\
[mesh]
name = "test-viewer"
token = "msh_testtoken{"a" * 60}"

[pipeline]
path = "./pipeline.py"

[source]
watch_dir = "/sources"
patterns = ["**/*.json"]

[server]
port = 7433

[cluster]
leader_candidates = []
"""
    (mesh_dir / "synix-mesh.toml").write_text(config_content)

    # Write state
    state = {
        "role": "server",
        "server_url": "",
        "my_hostname": "test-host",
        "term": {"counter": 1, "leader_id": "test-host"},
    }
    (mesh_dir / "state.json").write_text(json.dumps(state))

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    return MeshContext(
        name="test-viewer",
        config_path=mesh_dir / "synix-mesh.toml",
        mesh_dir=mesh_dir,
        build_dir=build_dir,
        state=state,
        server_url="",
        token="msh_test",
        role="server",
        console=console,
        pipeline_path="./pipeline.py",
    )


def _get_output(ctx: MeshContext) -> str:
    """Extract rendered text from the console buffer."""
    ctx.console.file.seek(0)
    return ctx.console.file.read()


class TestRenderOverview:
    def test_shows_memory_tree(self, mesh_ctx):
        render_overview(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "Memory Tree" in output
        assert "transcripts" in output
        assert "episodes" in output
        assert "core" in output

    def test_shows_layer_counts(self, mesh_ctx):
        render_overview(mesh_ctx)
        output = _get_output(mesh_ctx)
        # 3 transcripts, 2 episodes, 1 core
        assert "3" in output
        assert "2" in output

    def test_shows_cluster_info(self, mesh_ctx):
        render_overview(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "test-host" in output

    def test_no_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        render_overview(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "No artifacts yet" in output


class TestRenderArtifacts:
    def test_shows_all_layers(self, mesh_ctx):
        render_artifacts(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "transcripts" in output
        assert "episodes" in output
        assert "core" in output

    def test_shows_artifact_labels(self, mesh_ctx):
        render_artifacts(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "tx-conv-0" in output
        assert "ep-conv-0" in output
        assert "core-v1" in output

    def test_shows_numbered_rows(self, mesh_ctx):
        render_artifacts(mesh_ctx)
        output = _get_output(mesh_ctx)
        # Should have row numbers
        assert "1" in output

    def test_filter_by_layer(self, mesh_ctx):
        render_artifacts(mesh_ctx, layer_filter="episodes")
        output = _get_output(mesh_ctx)
        assert "episodes" in output
        assert "transcripts" not in output.split("episodes")[0]  # transcripts layer header shouldn't appear

    def test_filter_nonexistent_layer(self, mesh_ctx):
        render_artifacts(mesh_ctx, layer_filter="nonexistent")
        output = _get_output(mesh_ctx)
        assert "No artifacts found in layer" in output

    def test_returns_ordered_labels(self, mesh_ctx):
        labels = render_artifacts(mesh_ctx)
        assert isinstance(labels, list)
        assert len(labels) == 6  # 3 transcripts + 2 episodes + 1 core

    def test_no_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        render_artifacts(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "No build directory" in output


class TestRenderDetail:
    def test_shows_artifact_content(self, mesh_ctx):
        render_detail(mesh_ctx, label="core-v1")
        output = _get_output(mesh_ctx)
        assert "Core Memory" in output
        assert "user prefers Python" in output

    def test_shows_metadata_header(self, mesh_ctx):
        render_detail(mesh_ctx, label="core-v1")
        output = _get_output(mesh_ctx)
        assert "core" in output  # layer name
        assert "core-v1" in output  # label

    def test_shows_provenance(self, mesh_ctx):
        render_detail(mesh_ctx, label="core-v1")
        output = _get_output(mesh_ctx)
        assert "Provenance" in output
        assert "ep-conv-0" in output
        assert "ep-conv-1" in output

    def test_provenance_shows_parents_of_parents(self, mesh_ctx):
        render_detail(mesh_ctx, label="core-v1")
        output = _get_output(mesh_ctx)
        # core-v1 -> ep-conv-0 -> tx-conv-0
        assert "tx-conv-0" in output

    def test_not_found(self, mesh_ctx):
        render_detail(mesh_ctx, label="nonexistent-label")
        output = _get_output(mesh_ctx)
        assert "not found" in output

    def test_prefix_resolution(self, mesh_ctx):
        render_detail(mesh_ctx, label="core")
        output = _get_output(mesh_ctx)
        assert "Core Memory" in output


class TestRenderSearch:
    def test_empty_query(self, mesh_ctx):
        results = render_search(mesh_ctx, query="")
        output = _get_output(mesh_ctx)
        assert results == []
        assert "Usage" in output

    def test_no_server_no_index(self, mesh_ctx):
        results = render_search(mesh_ctx, query="python")
        output = _get_output(mesh_ctx)
        assert "No results" in output


class TestRenderConfig:
    def test_shows_sections(self, mesh_ctx):
        render_config(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "source" in output
        assert "server" in output
        assert "cluster" in output

    def test_shows_values(self, mesh_ctx):
        render_config(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "7433" in output
        assert "/sources" in output

    def test_missing_config(self, mesh_ctx, tmp_path):
        mesh_ctx.config_path = tmp_path / "nonexistent.toml"
        render_config(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "not found" in output


class TestRenderBuilds:
    def test_shows_artifact_counts(self, mesh_ctx):
        render_builds(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "Artifacts by Layer" in output
        assert "transcripts" in output
        assert "episodes" in output
        assert "core" in output

    def test_no_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        render_builds(mesh_ctx)
        output = _get_output(mesh_ctx)
        assert "No build directory" in output


class TestDispatch:
    def test_empty_returns_overview(self):
        vs = ViewState(view="artifacts")
        result = dispatch("", vs)
        assert result.view == "overview"

    def test_q_quits(self):
        vs = ViewState()
        result = dispatch("q", vs)
        assert result.view == "quit"

    def test_quit_quits(self):
        vs = ViewState()
        result = dispatch("quit", vs)
        assert result.view == "quit"

    def test_a_goes_to_artifacts(self):
        vs = ViewState()
        result = dispatch("a", vs)
        assert result.view == "artifacts"
        assert result.layer_filter == ""

    def test_a_with_layer_filter(self):
        vs = ViewState()
        result = dispatch("a episodes", vs)
        assert result.view == "artifacts"
        assert result.layer_filter == "episodes"

    def test_artifacts_full_word(self):
        vs = ViewState()
        result = dispatch("artifacts episodes", vs)
        assert result.view == "artifacts"
        assert result.layer_filter == "episodes"

    def test_s_goes_to_search(self):
        vs = ViewState()
        result = dispatch("s python memory", vs)
        assert result.view == "search"
        assert result.search_query == "python memory"

    def test_search_full_word(self):
        vs = ViewState()
        result = dispatch("search hello", vs)
        assert result.view == "search"
        assert result.search_query == "hello"

    def test_c_goes_to_config(self):
        vs = ViewState()
        result = dispatch("c", vs)
        assert result.view == "config"

    def test_b_goes_to_builds(self):
        vs = ViewState()
        result = dispatch("b", vs)
        assert result.view == "builds"

    def test_number_from_artifacts(self):
        vs = ViewState(
            view="artifacts",
            artifact_labels=["tx-conv-0", "tx-conv-1", "ep-conv-0"],
        )
        result = dispatch("2", vs)
        assert result.view == "detail"
        assert result.artifact_label == "tx-conv-1"

    def test_number_from_search(self):
        vs = ViewState(
            view="search",
            search_results=[
                {"label": "ep-conv-0"},
                {"label": "core-v1"},
            ],
        )
        result = dispatch("1", vs)
        assert result.view == "detail"
        assert result.artifact_label == "ep-conv-0"

    def test_number_out_of_range(self):
        vs = ViewState(view="artifacts", artifact_labels=["tx-conv-0"])
        result = dispatch("5", vs)
        # Should stay on current view (no crash)
        assert result.view == "artifacts"

    def test_exit_command(self):
        vs = ViewState()
        result = dispatch("exit", vs)
        assert result.view == "quit"

    def test_p_goes_to_pipeline(self):
        vs = ViewState()
        result = dispatch("p", vs)
        assert result.view == "pipeline"

    def test_pipeline_full_word(self):
        vs = ViewState()
        result = dispatch("pipeline", vs)
        assert result.view == "pipeline"


SAMPLE_PIPELINE = """\
from synix import Pipeline, Source, FlatFile, SearchIndex
from synix.transforms import EpisodeSummary, CoreSynthesis

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
core = CoreSynthesis("core", depends_on=[episodes])

search = SearchIndex("search", sources=[episodes, core])
context = FlatFile("context", sources=[core], output_path="./build/context.md")

pipeline = Pipeline(
    "test-pipeline",
    source_dir="./sources",
    build_dir="./build",
    concurrency=3,
    llm_config={"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
)
pipeline.add(transcripts, episodes, core, search, context)
"""


@pytest.fixture
def pipeline_ctx(tmp_path, build_dir):
    """Create a MeshContext with a real pipeline.py file."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()

    # Write the pipeline file into the mesh dir
    pipeline_file = mesh_dir / "pipeline.py"
    pipeline_file.write_text(SAMPLE_PIPELINE)

    config_content = f"""\
[mesh]
name = "test-viewer"
token = "msh_testtoken{"a" * 60}"

[pipeline]
path = "./pipeline.py"

[source]
watch_dir = "/sources"
patterns = ["**/*.json"]

[server]
port = 7433

[cluster]
leader_candidates = []
"""
    (mesh_dir / "synix-mesh.toml").write_text(config_content)

    state = {"role": "server", "server_url": "", "my_hostname": "test-host"}
    (mesh_dir / "state.json").write_text(json.dumps(state))

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    return MeshContext(
        name="test-viewer",
        config_path=mesh_dir / "synix-mesh.toml",
        mesh_dir=mesh_dir,
        build_dir=build_dir,
        state=state,
        server_url="",
        token="msh_test",
        role="server",
        console=console,
        pipeline_path="./pipeline.py",
    )


def _get_pipeline_output(ctx: MeshContext) -> str:
    """Extract rendered text from the console buffer."""
    ctx.console.file.seek(0)
    return ctx.console.file.read()


class TestRenderPipeline:
    def test_shows_pipeline_name(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "test-pipeline" in output

    def test_shows_layers_in_tree(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "transcripts" in output
        assert "episodes" in output
        assert "core" in output

    def test_shows_layer_types(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "Source" in output
        assert "EpisodeSummary" in output
        assert "CoreSynthesis" in output

    def test_shows_dependencies(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        # Dependency arrows
        assert "\u2190" in output

    def test_shows_projections(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "Projections" in output
        assert "SearchIndex" in output
        assert "FlatFile" in output
        assert "search" in output
        assert "context" in output

    def test_shows_metadata(self, pipeline_ctx):
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "Concurrency" in output
        assert "3" in output
        assert "claude-sonnet" in output

    def test_pipeline_not_found(self, pipeline_ctx):
        pipeline_ctx.pipeline_path = "./nonexistent.py"
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "not found" in output

    def test_no_pipeline_path(self, pipeline_ctx):
        pipeline_ctx.pipeline_path = ""
        render_pipeline(pipeline_ctx)
        output = _get_pipeline_output(pipeline_ctx)
        assert "No pipeline path" in output
