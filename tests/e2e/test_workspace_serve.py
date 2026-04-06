"""E2e tests for Workspace server integration.

Tests runtime activation, queue/prompt access through workspace,
and the serve-time wiring that replaces the old _state dict.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synix.workspace import (
    WorkspaceRuntime,
    WorkspaceState,
    init_workspace,
    open_workspace,
)


@pytest.fixture
def ws(tmp_path: Path):
    """A workspace with pipeline and source data, ready to build."""
    ws = init_workspace(tmp_path / "serve-test")
    (ws.root / "pipeline.py").write_text(
        textwrap.dedent("""\
        from synix import Pipeline, Source
        pipeline = Pipeline("serve-test", source_dir="./sources")
        pipeline.add(Source("notes"))
    """)
    )
    (ws.root / "sources" / "note.md").write_text("Test note for serve integration.")
    return ws


class TestRuntimeActivation:
    def test_activate_sets_serving_state(self, ws) -> None:
        assert ws.state != WorkspaceState.SERVING
        ws.activate_runtime(
            queue=MagicMock(),
            prompt_store=MagicMock(),
            build_lock=MagicMock(),
        )
        assert ws.state == WorkspaceState.SERVING

    def test_runtime_services_accessible(self, ws) -> None:
        mock_queue = MagicMock()
        mock_prompts = MagicMock()
        mock_lock = MagicMock()

        rt = ws.activate_runtime(
            queue=mock_queue,
            prompt_store=mock_prompts,
            build_lock=mock_lock,
        )

        assert isinstance(rt, WorkspaceRuntime)
        assert ws.runtime.queue is mock_queue
        assert ws.runtime.prompt_store is mock_prompts
        assert ws.runtime.build_lock is mock_lock

    def test_runtime_with_vllm_override(self, ws) -> None:
        override = {"provider": "openai-compatible", "model": "test", "base_url": "http://localhost:8100/v1"}
        ws.activate_runtime(
            queue=MagicMock(),
            prompt_store=MagicMock(),
            build_lock=MagicMock(),
            llm_config_override=override,
        )
        assert ws.runtime.llm_config_override == override


class TestWorkspaceWithRealQueue:
    def test_queue_through_workspace(self, ws) -> None:
        from synix.server.queue import DocumentQueue

        queue = DocumentQueue(ws.synix_dir / "queue.db")
        ws.activate_runtime(
            queue=queue,
            prompt_store=MagicMock(),
            build_lock=MagicMock(),
        )

        # Ingest through the queue
        doc_id = ws.runtime.queue.enqueue(
            "notes",
            "test.md",
            "abc123hash",
            str(ws.root / "sources" / "note.md"),
        )
        assert doc_id is not None
        assert ws.runtime.queue.pending_count() == 1

    def test_prompt_store_through_workspace(self, ws) -> None:
        from synix.server.prompt_store import PromptStore

        store = PromptStore(ws.synix_dir / "prompts.db")
        ws.activate_runtime(
            queue=MagicMock(),
            prompt_store=store,
            build_lock=MagicMock(),
        )

        ws.runtime.prompt_store.put("test-prompt", "Hello {name}!")
        assert ws.runtime.prompt_store.get("test-prompt") == "Hello {name}!"


class TestBuildThroughWorkspace:
    def test_build_and_release(self, ws, mock_llm) -> None:
        ws.load_pipeline()
        result = ws.build()
        assert result.built > 0

        ws.release_to("local")
        assert ws.state == WorkspaceState.RELEASED

        # Search works
        release = ws.release("local")
        arts = list(release.artifacts())
        assert len(arts) > 0

    def test_workspace_config_from_toml(self, ws) -> None:
        """Reopen workspace and verify config parsed from synix.toml."""
        ws2 = open_workspace(str(ws.root))
        assert ws2.name == "serve-test"
        assert ws2.config.pipeline_path == "pipeline.py"
