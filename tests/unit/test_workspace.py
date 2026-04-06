"""Tests for the Workspace abstraction."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import synix
from synix.workspace import (
    BucketConfig,
    Workspace,
    WorkspaceConfig,
    WorkspaceRuntime,
    WorkspaceState,
    _parse_toml,
    init_workspace,
    load_server_bindings,
    open_workspace,
)


@pytest.fixture
def project(tmp_path: Path):
    """A bare synix project."""
    return synix.init(str(tmp_path / "test-ws"))


@pytest.fixture
def workspace(project) -> Workspace:
    """A workspace wrapping a bare project."""
    return Workspace(project)


@pytest.fixture
def config() -> WorkspaceConfig:
    return WorkspaceConfig(
        name="test-workspace",
        pipeline_path="pipeline.py",
        buckets=[
            BucketConfig(name="docs", dir="sources/docs", patterns=["**/*.md"]),
            BucketConfig(name="sessions", dir="sources/sessions"),
        ],
    )


@pytest.fixture
def configured_workspace(project, config) -> Workspace:
    return Workspace(project, config)


# --- Identity ---


class TestIdentity:
    def test_name_from_config(self, configured_workspace: Workspace) -> None:
        assert configured_workspace.name == "test-workspace"

    def test_name_from_directory(self, workspace: Workspace) -> None:
        assert workspace.name == "test-ws"

    def test_root(self, workspace: Workspace) -> None:
        assert workspace.root.name == "test-ws"

    def test_synix_dir(self, workspace: Workspace) -> None:
        assert (workspace.synix_dir / "objects").is_dir()


# --- State ---


class TestState:
    def test_fresh(self, workspace: Workspace) -> None:
        assert workspace.state == WorkspaceState.FRESH

    def test_configured_with_buckets(self, configured_workspace: Workspace) -> None:
        assert configured_workspace.state == WorkspaceState.CONFIGURED

    def test_configured_with_pipeline(self, workspace: Workspace, tmp_path: Path) -> None:
        # Write a minimal pipeline
        pipeline_file = workspace.root / "pipeline.py"
        pipeline_file.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test")
            pipeline.add(Source("notes"))
        """)
        )
        workspace.load_pipeline()
        assert workspace.state == WorkspaceState.CONFIGURED

    def test_built(self, workspace: Workspace, mock_llm) -> None:
        pipeline_file = workspace.root / "pipeline.py"
        pipeline_file.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test", source_dir="./sources")
            pipeline.add(Source("notes"))
        """)
        )
        sources = workspace.root / "sources"
        sources.mkdir(exist_ok=True)
        (sources / "note.md").write_text("hello world")
        workspace.load_pipeline()
        workspace.build()
        assert workspace.state == WorkspaceState.BUILT

    def test_released(self, workspace: Workspace, mock_llm) -> None:
        pipeline_file = workspace.root / "pipeline.py"
        pipeline_file.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test", source_dir="./sources")
            pipeline.add(Source("notes"))
        """)
        )
        sources = workspace.root / "sources"
        sources.mkdir(exist_ok=True)
        (sources / "note.md").write_text("hello world")
        workspace.load_pipeline()
        workspace.build()
        workspace.release_to("local")
        assert workspace.state == WorkspaceState.RELEASED

    def test_serving(self, workspace: Workspace) -> None:
        from unittest.mock import MagicMock

        workspace.activate_runtime(
            queue=MagicMock(),
            prompt_store=MagicMock(),
            build_lock=MagicMock(),
        )
        assert workspace.state == WorkspaceState.SERVING


# --- Delegation ---


class TestDelegation:
    def test_project_accessible(self, workspace: Workspace) -> None:
        assert workspace.project is not None
        assert workspace.project.synix_dir == workspace.synix_dir

    def test_pipeline_none_initially(self, workspace: Workspace) -> None:
        assert workspace.pipeline is None

    def test_releases_empty_initially(self, workspace: Workspace) -> None:
        assert workspace.releases() == []


# --- Buckets ---


class TestBuckets:
    def test_buckets_from_config(self, configured_workspace: Workspace) -> None:
        assert len(configured_workspace.buckets) == 2
        assert configured_workspace.buckets[0].name == "docs"

    def test_bucket_dir_relative(self, configured_workspace: Workspace) -> None:
        d = configured_workspace.bucket_dir("docs")
        assert d == configured_workspace.root / "sources" / "docs"

    def test_bucket_dir_absolute(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "absolute-bucket")
        config = WorkspaceConfig(buckets=[BucketConfig(name="abs", dir=abs_path)])
        import synix

        project = synix.init(str(tmp_path / "abs-bucket-ws"))
        ws = Workspace(project, config)
        assert ws.bucket_dir("abs") == Path(abs_path)

    def test_bucket_dir_not_found(self, configured_workspace: Workspace) -> None:
        with pytest.raises(ValueError, match="not found"):
            configured_workspace.bucket_dir("nonexistent")

    def test_buckets_empty_no_config(self, workspace: Workspace) -> None:
        assert workspace.buckets == []


# --- Runtime ---


class TestRuntime:
    def test_runtime_none_initially(self, workspace: Workspace) -> None:
        assert workspace.runtime is None

    def test_activate_runtime(self, workspace: Workspace) -> None:
        from unittest.mock import MagicMock

        rt = workspace.activate_runtime(
            queue=MagicMock(),
            prompt_store=MagicMock(),
            build_lock=MagicMock(),
        )
        assert isinstance(rt, WorkspaceRuntime)
        assert workspace.runtime is rt


# --- TOML parsing ---


class TestConfigParsing:
    def test_parse_full_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / "synix.toml"
        toml.write_text(
            textwrap.dedent("""\
            [workspace]
            name = "my-memory"
            pipeline_path = "my_pipeline.py"

            [buckets.sessions]
            dir = "sources/sessions"
            patterns = ["**/*.jsonl.gz"]
            description = "Session transcripts"

            [buckets.docs]
            dir = "sources/docs"

            [auto_build]
            enabled = true
            window = 60

            [vllm]
            enabled = true
            model = "Qwen/Qwen3.5-2B"
            port = 8100
        """)
        )

        config = _parse_toml(toml, tmp_path)
        assert config.name == "my-memory"
        assert config.pipeline_path == "my_pipeline.py"
        assert len(config.buckets) == 2
        assert config.auto_build.window == 60
        assert config.vllm.enabled is True
        assert config.vllm.model == "Qwen/Qwen3.5-2B"

    def test_parse_minimal_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / "synix.toml"
        toml.write_text('[workspace]\nname = "bare"\n')

        config = _parse_toml(toml, tmp_path)
        assert config.name == "bare"
        assert config.buckets == []
        assert config.vllm.enabled is False

    def test_parse_defaults_from_directory(self, tmp_path: Path) -> None:
        toml = tmp_path / "synix.toml"
        toml.write_text("[workspace]\n")

        config = _parse_toml(toml, tmp_path)
        assert config.name == tmp_path.name
        assert config.pipeline_path == "pipeline.py"

    def test_server_bindings(self, tmp_path: Path) -> None:
        toml = tmp_path / "synix.toml"
        toml.write_text(
            textwrap.dedent("""\
            [workspace]
            name = "test"

            [server]
            mcp_port = 9000
            viewer_port = 9001
        """)
        )

        bindings = load_server_bindings(str(toml))
        assert bindings.mcp_port == 9000
        assert bindings.viewer_port == 9001

    def test_server_bindings_defaults(self, tmp_path: Path) -> None:
        bindings = load_server_bindings(str(tmp_path / "nonexistent.toml"))
        assert bindings.mcp_port == 8200
        assert bindings.viewer_port == 9471


# --- Factories ---


class TestFactories:
    def test_open_workspace_bare(self, tmp_path: Path) -> None:
        synix.init(str(tmp_path / "bare-ws"))
        ws = open_workspace(str(tmp_path / "bare-ws"))
        assert ws.name == "bare-ws"
        assert ws.state == WorkspaceState.FRESH
        assert ws.config is not None

    def test_open_workspace_with_toml(self, tmp_path: Path) -> None:
        synix.init(str(tmp_path / "configured-ws"))
        (tmp_path / "configured-ws" / "synix.toml").write_text('[workspace]\nname = "my-configured"\n')
        ws = open_workspace(str(tmp_path / "configured-ws"))
        assert ws.name == "my-configured"

    def test_init_workspace_creates_toml(self, tmp_path: Path) -> None:
        ws = init_workspace(tmp_path / "new-ws")
        assert (ws.root / "synix.toml").exists()
        assert ws.name == "new-ws"

    def test_init_workspace_preserves_existing_toml(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "existing-ws"
        ws_dir.mkdir()
        (ws_dir / "synix.toml").write_text('[workspace]\nname = "custom-name"\n')
        ws = init_workspace(ws_dir)
        assert ws.name == "custom-name"

    def test_open_workspace_explicit_config(self, tmp_path: Path) -> None:
        synix.init(str(tmp_path / "ws"))
        config_file = tmp_path / "custom-config.toml"
        config_file.write_text('[workspace]\nname = "from-custom"\n')
        ws = open_workspace(str(tmp_path / "ws"), config_path=str(config_file))
        assert ws.name == "from-custom"

    def test_open_workspace_missing_config_raises(self, tmp_path: Path) -> None:
        synix.init(str(tmp_path / "ws"))
        with pytest.raises(FileNotFoundError):
            open_workspace(str(tmp_path / "ws"), config_path="/nonexistent/config.toml")
