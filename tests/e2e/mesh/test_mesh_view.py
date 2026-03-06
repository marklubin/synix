"""E2E: Interactive memory viewer — Textual TUI tests using async pilot."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console
from textual.widgets import Markdown

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.core.models import Artifact
from synix.mesh.viewer import (
    DetailScreen,
    MeshContext,
    MeshViewerApp,
    SearchScreen,
)

pytestmark = pytest.mark.asyncio


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

    state = {
        "role": "server",
        "server_url": "",
        "my_hostname": "test-host",
        "term": {"counter": 1, "leader_id": "test-host"},
    }
    (mesh_dir / "state.json").write_text(json.dumps(state))

    return MeshContext(
        name="test-viewer",
        config_path=mesh_dir / "synix-mesh.toml",
        mesh_dir=mesh_dir,
        build_dir=build_dir,
        state=state,
        server_url="",
        token="msh_test",
        role="server",
        console=Console(file=StringIO(), force_terminal=True, width=120),
        pipeline_path="./pipeline.py",
    )


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

    return MeshContext(
        name="test-viewer",
        config_path=mesh_dir / "synix-mesh.toml",
        mesh_dir=mesh_dir,
        build_dir=build_dir,
        state=state,
        server_url="",
        token="msh_test",
        role="server",
        console=Console(file=StringIO(), force_terminal=True, width=120),
        pipeline_path="./pipeline.py",
    )


def _render_static_to_text(static) -> str:
    """Render a Static widget's content to plain text for assertions."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, no_color=True)
    console.print(static.content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------


class TestOverviewTab:
    async def test_shows_memory_tree(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#overview-content"))
            assert "Memory Tree" in content
            assert "transcripts" in content
            assert "episodes" in content
            assert "core" in content

    async def test_shows_layer_counts(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#overview-content"))
            # 3 transcripts, 2 episodes, 1 core
            assert "3" in content
            assert "2" in content

    async def test_shows_cluster_info(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#overview-content"))
            assert "test-host" in content

    async def test_no_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#overview-content"))
            assert "No artifacts yet" in content


# ---------------------------------------------------------------------------
# Artifacts tab
# ---------------------------------------------------------------------------


class TestArtifactsTab:
    async def test_table_has_correct_row_count(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import DataTable

            table = app.query_one("#artifacts-table", DataTable)
            assert table.row_count == 6  # 3 tx + 2 ep + 1 core

    async def test_table_shows_layer_names(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import DataTable

            table = app.query_one("#artifacts-table", DataTable)
            # Collect all Layer column values (column index 1)
            layers = set()
            for row_key in table.rows:
                row_data = table.get_row(row_key)
                layers.add(str(row_data[1]))
            assert "transcripts" in layers
            assert "episodes" in layers
            assert "core" in layers

    async def test_table_shows_labels(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import DataTable

            table = app.query_one("#artifacts-table", DataTable)
            labels = set()
            for row_key in table.rows:
                row_data = table.get_row(row_key)
                labels.add(str(row_data[2]))
            assert "tx-conv-0" in labels
            assert "ep-conv-0" in labels
            assert "core-v1" in labels

    async def test_enter_pushes_detail_screen(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Switch to artifacts tab and focus the table
            from textual.widgets import DataTable, TabbedContent

            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-artifacts"
            await pilot.pause()

            table = app.query_one("#artifacts-table", DataTable)
            table.focus()
            await pilot.pause()

            # Move cursor to first row and press enter
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, DetailScreen)

    async def test_empty_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import DataTable

            table = app.query_one("#artifacts-table", DataTable)
            assert table.row_count == 0

            empty_msg = app.query_one("#artifacts-empty")
            assert "No build directory" in str(empty_msg.content)

    async def test_empty_manifest(self, tmp_path):
        """Build dir exists but has no artifacts."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        ArtifactStore(build_dir)  # Creates manifest file

        ctx = MeshContext(
            name="test",
            config_path=tmp_path / "config.toml",
            mesh_dir=tmp_path,
            build_dir=build_dir,
            state={},
            server_url="",
            token="",
            role="server",
            console=Console(file=StringIO(), width=120),
        )

        app = MeshViewerApp(ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import DataTable

            table = app.query_one("#artifacts-table", DataTable)
            assert table.row_count == 0

            empty_msg = app.query_one("#artifacts-empty")
            assert "No artifacts" in str(empty_msg.content)


# ---------------------------------------------------------------------------
# Detail screen
# ---------------------------------------------------------------------------


class TestDetailScreen:
    async def test_shows_artifact_content(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core-v1"))
            await pilot.pause()

            md = app.screen.query_one("#detail-markdown", Markdown)
            # Textual Markdown stores the raw source
            assert "Core Memory" in md._markdown or "user prefers Python" in md._markdown

    async def test_shows_metadata(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core-v1"))
            await pilot.pause()

            meta = _render_static_to_text(app.screen.query_one("#detail-meta"))
            assert "core" in meta
            assert "core-v1" in meta

    async def test_shows_provenance(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core-v1"))
            await pilot.pause()

            prov = _render_static_to_text(app.screen.query_one("#detail-provenance"))
            assert "Provenance" in prov
            assert "ep-conv-0" in prov
            assert "ep-conv-1" in prov

    async def test_provenance_shows_grandparents(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core-v1"))
            await pilot.pause()

            prov = _render_static_to_text(app.screen.query_one("#detail-provenance"))
            # core-v1 -> ep-conv-0 -> tx-conv-0
            assert "tx-conv-0" in prov

    async def test_not_found(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "nonexistent-label"))
            await pilot.pause()

            error = app.screen.query_one("#detail-error")
            assert "not found" in str(error.content)

    async def test_prefix_resolution(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core"))
            await pilot.pause()

            md = app.screen.query_one("#detail-markdown", Markdown)
            assert "Core Memory" in md._markdown

    async def test_escape_pops_back(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(DetailScreen(mesh_ctx, "core-v1"))
            await pilot.pause()
            assert isinstance(app.screen, DetailScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, DetailScreen)


# ---------------------------------------------------------------------------
# Pipeline tab
# ---------------------------------------------------------------------------


class TestPipelineTab:
    async def test_shows_pipeline_name(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "test-pipeline" in content

    async def test_shows_layers(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "transcripts" in content
            assert "episodes" in content
            assert "core" in content

    async def test_shows_layer_types(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "Source" in content
            assert "EpisodeSummary" in content
            assert "CoreSynthesis" in content

    async def test_shows_dependencies(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "\u2190" in content

    async def test_shows_projections(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "Projections" in content
            assert "SearchIndex" in content
            assert "FlatFile" in content

    async def test_shows_metadata(self, pipeline_ctx):
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "Concurrency" in content
            assert "3" in content
            assert "claude-sonnet" in content

    async def test_pipeline_not_found(self, pipeline_ctx):
        pipeline_ctx.pipeline_path = "./nonexistent.py"
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "not found" in content

    async def test_no_pipeline_path(self, pipeline_ctx):
        pipeline_ctx.pipeline_path = ""
        app = MeshViewerApp(pipeline_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#pipeline-content"))
            assert "No pipeline path" in content


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


class TestConfigTab:
    async def test_shows_sections(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#config-content"))
            assert "source" in content
            assert "server" in content
            assert "cluster" in content

    async def test_shows_values(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#config-content"))
            assert "7433" in content
            assert "/sources" in content

    async def test_missing_config(self, mesh_ctx, tmp_path):
        mesh_ctx.config_path = tmp_path / "nonexistent.toml"
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#config-content"))
            assert "not found" in content


# ---------------------------------------------------------------------------
# Builds tab
# ---------------------------------------------------------------------------


class TestBuildsTab:
    async def test_shows_artifact_counts(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#builds-content"))
            assert "Artifacts by Layer" in content
            assert "transcripts" in content
            assert "episodes" in content
            assert "core" in content

    async def test_no_build_dir(self, mesh_ctx, tmp_path):
        mesh_ctx.build_dir = tmp_path / "nonexistent"
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            content = _render_static_to_text(app.query_one("#builds-content"))
            assert "No build directory" in content


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_slash_pushes_search_screen(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            assert isinstance(app.screen, SearchScreen)

    async def test_search_input_focused(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import Input

            search_input = app.screen.query_one("#search-input", Input)
            assert search_input.has_focus

    async def test_search_shows_results_via_manifest(self, mesh_ctx):
        """Search finds artifacts via manifest scan (no server, no search.db)."""
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import DataTable, Input

            search_input = app.screen.query_one("#search-input", Input)
            search_input.value = "Episode"
            await pilot.press("enter")
            await pilot.pause()

            results_table = app.screen.query_one("#search-results", DataTable)
            assert results_table.row_count >= 2  # At least 2 episodes match

            # Status should show results count
            status = app.screen.query_one("#search-status")
            assert "result" in str(status.content)

    async def test_search_shows_results_via_mock(self, mesh_ctx, monkeypatch):
        """Search works when local_search returns results."""
        monkeypatch.setattr(
            "synix.mesh.viewer._local_search",
            lambda ctx, q: [
                {
                    "label": "ep-conv-0",
                    "layer_name": "episodes",
                    "layer_level": 1,
                    "score": 0.95,
                    "content": "Episode 0 summary content",
                },
            ],
        )
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import DataTable, Input

            search_input = app.screen.query_one("#search-input", Input)
            search_input.value = "test query"
            await pilot.press("enter")
            await pilot.pause()

            results_table = app.screen.query_one("#search-results", DataTable)
            assert results_table.row_count == 1

    async def test_search_no_results(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import DataTable, Input

            search_input = app.screen.query_one("#search-input", Input)
            search_input.value = "zzz_xyzzy_nonexistent_999"
            await pilot.press("enter")
            await pilot.pause()

            results_table = app.screen.query_one("#search-results", DataTable)
            assert results_table.row_count == 0

            # Should show "no results" status
            status = app.screen.query_one("#search-status")
            assert "No results" in str(status.content)

    async def test_escape_from_search(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            assert isinstance(app.screen, SearchScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, SearchScreen)

    async def test_search_result_opens_detail(self, mesh_ctx):
        """Selecting a search result pushes the detail screen."""
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import DataTable, Input

            # Search for "core" — should match via manifest scan
            search_input = app.screen.query_one("#search-input", Input)
            search_input.value = "core"
            await pilot.press("enter")
            await pilot.pause()

            results_table = app.screen.query_one("#search-results", DataTable)
            assert results_table.row_count >= 1

            results_table.focus()
            await pilot.pause()
            results_table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, DetailScreen)


# ---------------------------------------------------------------------------
# Quit
# ---------------------------------------------------------------------------


class TestQuit:
    async def test_q_quits(self, mesh_ctx):
        app = MeshViewerApp(mesh_ctx)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            # App should be exiting — the run_test context exits cleanly
