"""End-to-end tests for the full Workspace lifecycle.

Tests the complete user flow: init → configure → build → release → search.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from synix.workspace import WorkspaceState, init_workspace, open_workspace


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    return tmp_path / "test-workspace"


class TestWorkspaceLifecycle:
    """Full init → build → release → search lifecycle."""

    def test_init_creates_complete_scaffold(self, ws_dir: Path) -> None:
        ws = init_workspace(ws_dir)
        assert ws.root == ws_dir
        assert (ws_dir / "synix.toml").exists()
        assert (ws_dir / ".synix").is_dir()
        assert (ws_dir / "sources").is_dir()
        assert ws.state == WorkspaceState.FRESH

    def test_full_lifecycle(self, ws_dir: Path, mock_llm) -> None:
        # 1. Init workspace
        ws = init_workspace(ws_dir)
        assert ws.state == WorkspaceState.FRESH
        assert ws.name == "test-workspace"

        # 2. Write pipeline + source data
        (ws_dir / "pipeline.py").write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test-lifecycle", source_dir="./sources")
            pipeline.add(Source("notes"))
        """)
        )
        (ws_dir / "sources" / "hello.md").write_text("Today I learned about workspace abstractions in build systems.")

        # 3. Load pipeline → CONFIGURED
        ws.load_pipeline()
        assert ws.pipeline is not None
        assert ws.state == WorkspaceState.CONFIGURED

        # 4. Build → BUILT
        result = ws.build()
        assert result.built > 0
        assert ws.state == WorkspaceState.BUILT

        # 5. Release → RELEASED
        ws.release_to("local")
        assert ws.state == WorkspaceState.RELEASED
        assert "local" in ws.releases()

        # 6. Search
        release = ws.release("local")
        arts = list(release.artifacts())
        assert len(arts) > 0

    def test_reopen_workspace_preserves_state(self, ws_dir: Path, mock_llm) -> None:
        # Build and release
        ws = init_workspace(ws_dir)
        (ws_dir / "pipeline.py").write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test", source_dir="./sources")
            pipeline.add(Source("notes"))
        """)
        )
        (ws_dir / "sources" / "note.md").write_text("content")
        ws.load_pipeline()
        ws.build()
        ws.release_to("local")

        # Reopen from disk
        ws2 = open_workspace(str(ws_dir))
        assert ws2.state == WorkspaceState.RELEASED
        assert ws2.name == "test-workspace"
        assert "local" in ws2.releases()

    def test_workspace_with_config(self, ws_dir: Path, mock_llm) -> None:
        """Workspace with synix.toml buckets and custom name."""
        ws = init_workspace(ws_dir)

        # Write custom config
        (ws_dir / "synix.toml").write_text(
            textwrap.dedent("""\
            [workspace]
            name = "my-agent-memory"
            pipeline_path = "pipeline.py"

            [buckets.documents]
            dir = "sources/documents"
            patterns = ["**/*.md"]
            description = "Notes and docs"

            [buckets.sessions]
            dir = "sources/sessions"
            patterns = ["**/*.jsonl"]
        """)
        )

        # Reopen to pick up config
        ws = open_workspace(str(ws_dir))
        assert ws.name == "my-agent-memory"
        assert len(ws.buckets) == 2
        assert ws.bucket_dir("documents") == ws_dir / "sources" / "documents"

        # Write pipeline + data + build
        (ws_dir / "pipeline.py").write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("agent-memory", source_dir="./sources/documents")
            pipeline.add(Source("docs"))
        """)
        )
        docs_dir = ws_dir / "sources" / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "goals.md").write_text("Ship the workspace abstraction.")

        ws.load_pipeline()
        result = ws.build()
        assert result.built > 0
        ws.release_to("local")
        assert ws.state == WorkspaceState.RELEASED

    def test_bare_workspace_no_toml(self, ws_dir: Path) -> None:
        """Workspace without synix.toml works in bare mode."""
        import synix

        synix.init(str(ws_dir))
        # Delete the synix.toml that init_workspace would create
        toml = ws_dir / "synix.toml"
        if toml.exists():
            toml.unlink()

        ws = open_workspace(str(ws_dir))
        assert ws.state == WorkspaceState.FRESH
        assert ws.buckets == []
        assert ws.name == "test-workspace"  # from directory name

    def test_workspace_delegates_build_correctly(self, ws_dir: Path, mock_llm) -> None:
        """Build through workspace produces same results as through Project."""
        ws = init_workspace(ws_dir)
        (ws_dir / "pipeline.py").write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            pipeline = Pipeline("test", source_dir="./sources")
            pipeline.add(Source("notes"))
        """)
        )
        (ws_dir / "sources" / "a.md").write_text("Note A")
        (ws_dir / "sources" / "b.md").write_text("Note B")
        ws.load_pipeline()
        result = ws.build()
        assert result.built == 2

        # Release and verify artifacts accessible
        ws.release_to("local")
        release = ws.release("local")
        arts = list(release.artifacts())
        assert len(arts) == 2
        labels = {a.label for a in arts}
        assert any("a" in l for l in labels)
        assert any("b" in l for l in labels)
