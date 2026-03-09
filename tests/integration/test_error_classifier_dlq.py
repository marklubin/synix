"""Integration tests — error classifier and DLQ in the build runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.error_classifier import LLMErrorClassifier
from synix.build.runner import run
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "exports"
    src.mkdir()
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", src / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", src / "claude_export.json")
    return src


@pytest.fixture
def build_dir(tmp_path):
    return tmp_path / "build"


@pytest.fixture
def pipeline_obj(build_dir):
    p = Pipeline("test-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    monthly = MonthlyRollup("monthly", depends_on=[episodes])
    core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

    p.add(transcripts, episodes, monthly, core)
    p.add(SearchIndex("memory-index", sources=[episodes, monthly, core], search=["fulltext"]))
    p.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))
    return p


class TestContentFilterDLQ:
    def test_content_filter_skipped_build_continues(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """A content filter error on one episode doesn't abort the build."""
        call_count = [0]
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            call_count[0] += 1
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            # Make the FIRST episode call hit a content filter
            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError(
                        "LLM API error processing episode ep-test: Error code: 400 - "
                        "{'error': {'code': 400, 'message': 'The request was rejected "
                        "because it was considered high risk', 'type': 'content_filter'}}"
                    )
                return MockResponse(
                    "This is a summary of the conversation. The user discussed "
                    "technical topics including programming concepts."
                )
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("In this month, the main themes were technical learning.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nMark is a software engineer.\n\n## Current Focus\nBuilding systems.")
            return MockResponse("Mock response.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        # Build should complete despite the content filter (with DLQ enabled)
        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        # DLQ should have exactly 1 entry
        assert len(result.dlq) == 1
        assert "content_filter" in result.dlq.entries[0].error_message
        assert result.dlq.entries[0].layer_name == "episodes"

        # Build should still have produced artifacts (all except the filtered one)
        assert result.built > 0

    def test_default_no_dlq_content_filter_is_fatal(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """Without --dlq, content filter errors abort the build (fail-closed default)."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        # Without error_classifier, content filter should be fatal
        with pytest.raises(RuntimeError, match="content_filter"):
            run(pipeline_obj, source_dir=str(source_dir))

    def test_auth_error_still_fatal(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """Auth errors (401) are classified as fatal and still abort the build."""

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            raise RuntimeError(
                "LLM API error: Error code: 401 - {'error': {'message': "
                "'The API Key appears to be invalid', 'type': 'invalid_authentication_error'}}"
            )

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        with pytest.raises(RuntimeError, match="invalid"):
            run(pipeline_obj, source_dir=str(source_dir))

    def test_dlq_summary_in_result(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """DLQ summary is available in the RunResult."""
        filter_indices = {0, 2}  # Skip first and third episode calls
        episode_call = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                idx = episode_call[0]
                episode_call[0] += 1
                if idx in filter_indices:
                    raise RuntimeError(
                        "Error code: 400 - {'error': {'message': 'high risk', 'type': 'content_filter'}}"
                    )
                return MockResponse("Episode summary of the conversation.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.\n\n## Focus\nSystems.")
            return MockResponse("Mock response.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        assert len(result.dlq) == 2
        assert "2 artifacts skipped" in result.dlq.summary()
        assert "RuntimeError" in result.dlq.summary()

    def test_dlq_entries_in_jsonl_log(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """DLQ entries are written to the JSONL structured log."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        # Find the JSONL log file — synix_dir is sibling to build_dir
        logs_dir = build_dir.parent / ".synix" / "logs"
        assert logs_dir.exists(), "Logs directory should exist"
        log_files = list(logs_dir.glob("*.jsonl"))
        assert len(log_files) >= 1, "At least one JSONL log file should exist"

        # Parse the log and find DLQ events
        log_path = log_files[-1]
        events = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        dlq_events = [e for e in events if e.get("event") == "artifact_dlq"]
        assert len(dlq_events) == 1
        assert dlq_events[0]["error_type"] == "RuntimeError"
        assert "content_filter" in dlq_events[0]["error_message"]
        assert dlq_events[0]["layer"] == "episodes"

        # Also check DLQ entries in run_finish event
        run_finish = [e for e in events if e.get("event") == "run_finish"]
        assert len(run_finish) == 1

    def test_sequential_content_filter_skipped(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """Content filter in sequential (concurrency=1) mode also uses DLQ."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), concurrency=1, error_classifier=LLMErrorClassifier())

        assert len(result.dlq) == 1
        assert result.built > 0

    def test_dlq_count_in_layer_stats(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """LayerStats.dlq_count is correctly incremented for affected layers."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        # Find episodes layer stats
        episode_stats = next(s for s in result.layer_stats if s.name == "episodes")
        assert episode_stats.dlq_count == 1

        # Other layers should have 0 DLQ
        for s in result.layer_stats:
            if s.name != "episodes":
                assert s.dlq_count == 0, f"layer {s.name} should have 0 dlq_count"

    def test_dlq_persisted_in_manifest(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """DLQ entries are persisted in the snapshot manifest for post-build inspection."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        assert result.manifest_oid is not None

        # Read the manifest from the object store
        from synix.build.object_store import ObjectStore
        from synix.build.refs import synix_dir_for_build_dir

        synix_dir = synix_dir_for_build_dir(build_dir)
        store = ObjectStore(synix_dir)
        manifest = store.get_json(result.manifest_oid)

        # Manifest should contain DLQ entries
        assert "dlq" in manifest, "Manifest should contain DLQ entries"
        assert len(manifest["dlq"]) == 1
        assert manifest["dlq"][0]["layer_name"] == "episodes"
        assert "content_filter" in manifest["dlq"][0]["error_message"]

    def test_downstream_layers_still_build_with_reduced_inputs(self, pipeline_obj, source_dir, build_dir, monkeypatch):
        """Downstream layers proceed with available inputs when upstream artifacts are DLQ'd."""
        filter_count = [0]

        class MockResponse:
            def __init__(self, text):
                self.content = [MagicMock(text=text)]
                self.model = "claude-sonnet-4-20250514"
                self.usage = MagicMock(input_tokens=100, output_tokens=50)

        def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0].get("content", "") if messages else ""

            if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
                if filter_count[0] == 0:
                    filter_count[0] += 1
                    raise RuntimeError("Error code: 400 - content_filter rejected prompt")
                return MockResponse("Episode summary.")
            elif "monthly" in content.lower() or "monthly overview" in content.lower():
                return MockResponse("Monthly overview of remaining episodes.")
            elif "core memory" in content.lower():
                return MockResponse("## Identity\nEngineer.\n\n## Focus\nSystems.")
            return MockResponse("Mock.")

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        result = run(pipeline_obj, source_dir=str(source_dir), error_classifier=LLMErrorClassifier())

        # 1 episode DLQ'd, but monthly/core should still build
        assert len(result.dlq) == 1

        # Monthly and core layers should have built artifacts
        monthly_stats = next(s for s in result.layer_stats if s.name == "monthly")
        core_stats = next(s for s in result.layer_stats if s.name == "core")
        assert monthly_stats.built > 0, "Monthly should build from available episodes"
        assert core_stats.built > 0, "Core should build from available monthly rollups"

        # Snapshot should be committed (build is considered successful)
        assert result.snapshot_oid is not None
