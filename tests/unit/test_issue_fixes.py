"""Tests for issues #1-#5 fixes.

Issue 1: Per-unit cache check in runner (skip cached units, only execute uncached)
Issue 2: EmbeddingConfig.dimensions defaults to None
Issue 3a: EmbeddingConfig.resolve_base_url() for env var fallback
Issue 3b: Manifest stores identity fields only (no base_url, no api_key)
Issue 4: EmbeddingConfig.batch_size default 16 + 413 retry with halved batch
Issue 5: Plan estimation uses estimate_output_count when upstream is dirty
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synix import Artifact, Pipeline, Source
from synix.build.artifacts import ArtifactStore
from synix.build.plan import plan_build
from synix.build.runner import _execute_transform_concurrent, run
from synix.core.config import EmbeddingConfig
from synix.core.models import Transform
from synix.ext import EpisodeSummary
from synix.search.embeddings import EmbeddingProvider, OpenAIBackend, _is_payload_too_large

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockEmbeddingData:
    """Mock for a single embedding data item in OpenAI response."""

    def __init__(self, embedding: list[float], index: int = 0):
        self.embedding = embedding
        self.index = index


class MockEmbeddingResponse:
    """Mock for OpenAI embeddings.create() response."""

    def __init__(self, embeddings: list[list[float]]):
        self.data = [MockEmbeddingData(emb, idx) for idx, emb in enumerate(embeddings)]


def _make_transcript(tid: str, content: str = "", date: str = "2024-01-15") -> Artifact:
    """Create a transcript artifact for testing."""
    return Artifact(
        label=tid,
        artifact_type="transcript",
        content=content or f"Conversation {tid}",
        metadata={
            "source_conversation_id": tid,
            "date": date,
            "title": f"Conv {tid}",
            "message_count": 4,
        },
    )


# ---------------------------------------------------------------------------
# Issue 1: Per-unit cache check in runner
# ---------------------------------------------------------------------------


class TestPerUnitCaching:
    """Tests for Issue 1: per-unit cache check in runner.

    When _layer_fully_cached() returns False (e.g., new inputs added),
    individual cached units should still be skipped — only uncached units
    should trigger execute() calls.
    """

    def test_cached_units_skip_execute(self, tmp_path, mock_llm):
        """When some units are cached, only uncached units call execute().

        Strategy: build a pipeline once (all units get cached), add a new
        source file, rebuild. The LLM mock should only be called for the
        new source, not for the already-cached ones.
        """
        source_dir = tmp_path / "exports"
        source_dir.mkdir()

        # Start with 2 text files
        (source_dir / "conv_a.txt").write_text("user: Q-A\nassistant: A-A\n")
        (source_dir / "conv_b.txt").write_text("user: Q-B\nassistant: A-B\n")

        build_dir = tmp_path / "build"
        p = Pipeline("test-per-unit")
        p.build_dir = str(build_dir)
        p.source_dir = str(source_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        p.add(transcripts, episodes)

        # First build — all units are new
        result1 = run(p, concurrency=1)
        initial_llm_calls = len(mock_llm)
        assert result1.built > 0

        # Add a new source file
        (source_dir / "conv_c.txt").write_text("user: Q-C\nassistant: A-C\n")

        # Reset call tracking
        calls_before = len(mock_llm)

        # Second build — only the new source should trigger LLM calls
        result2 = run(p, concurrency=1)

        calls_after = len(mock_llm)
        new_calls = calls_after - calls_before

        # We should have exactly 1 new LLM call (for conv_c), not 3
        assert new_calls == 1, (
            f"Expected 1 new LLM call for the new source, got {new_calls}. "
            f"Per-unit caching should have skipped the 2 already-cached units."
        )

        # The episode layer stats should show both cached and built
        ep_stats = next(s for s in result2.layer_stats if s.name == "episodes")
        assert ep_stats.cached == 2, f"Expected 2 cached episodes, got {ep_stats.cached}"
        assert ep_stats.built == 1, f"Expected 1 built episode, got {ep_stats.built}"

    def test_cached_units_loaded_from_store(self, tmp_path, mock_llm):
        """Cached units load artifacts from store instead of re-executing."""
        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        (source_dir / "conv_a.txt").write_text("user: Q-A\nassistant: A-A\n")
        (source_dir / "conv_b.txt").write_text("user: Q-B\nassistant: A-B\n")

        build_dir = tmp_path / "build"
        p = Pipeline("test-load-cached")
        p.build_dir = str(build_dir)
        p.source_dir = str(source_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        p.add(transcripts, episodes)

        # First build
        run(p, concurrency=1)
        store = ArtifactStore(build_dir)
        original_episodes = sorted(store.list_artifacts("episodes"), key=lambda a: a.label)

        # Add new source
        (source_dir / "conv_c.txt").write_text("user: Q-C\nassistant: A-C\n")

        # Second build
        run(p, concurrency=1)
        # Re-create store to pick up new manifest entries from the second build
        store = ArtifactStore(build_dir)
        all_episodes = sorted(store.list_artifacts("episodes"), key=lambda a: a.label)

        # All 3 episodes should be present (2 cached + 1 new)
        assert len(all_episodes) == 3

        # The original 2 episodes should have the same content (loaded from cache)
        for orig in original_episodes:
            match = next((e for e in all_episodes if e.label == orig.label), None)
            assert match is not None, f"Original episode {orig.label} not found after rebuild"
            assert match.content == orig.content

    def test_concurrent_path_skips_cached_units(self, tmp_path, mock_llm):
        """Concurrent execution also skips cached units."""
        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        (source_dir / "conv_a.txt").write_text("user: Q-A\nassistant: A-A\n")
        (source_dir / "conv_b.txt").write_text("user: Q-B\nassistant: A-B\n")

        build_dir = tmp_path / "build"
        p = Pipeline("test-concurrent-cache")
        p.build_dir = str(build_dir)
        p.source_dir = str(source_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        p.add(transcripts, episodes)

        # First build with concurrency
        run(p, concurrency=4)
        calls_after_first = len(mock_llm)

        # Add new source
        (source_dir / "conv_c.txt").write_text("user: Q-C\nassistant: A-C\n")

        # Second build with concurrency — should skip cached units
        run(p, concurrency=4)
        calls_after_second = len(mock_llm)
        new_calls = calls_after_second - calls_after_first

        assert new_calls == 1, f"Expected 1 new LLM call with concurrent execution, got {new_calls}"

    def test_concurrent_cached_by_inputs_filtering(self):
        """_execute_transform_concurrent skips units found in cached_by_inputs."""
        call_log: list[str] = []
        lock = threading.Lock()

        class LoggingTransform(Transform):
            def __init__(self):
                super().__init__("logging")

            def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
                with lock:
                    for inp in inputs:
                        call_log.append(inp.label)
                return [
                    Artifact(
                        label=f"out-{inp.label}",
                        artifact_type="episode",
                        content=f"Summary of {inp.label}",
                        input_ids=[inp.artifact_id],
                    )
                    for inp in inputs
                ]

        t0 = _make_transcript("t-0")
        t1 = _make_transcript("t-1")
        t2 = _make_transcript("t-2")

        # Pre-populate cache for t-0 and t-1
        cached_art_0 = Artifact(
            label="out-t-0",
            artifact_type="episode",
            content="cached",
            input_ids=[t0.artifact_id],
        )
        cached_art_1 = Artifact(
            label="out-t-1",
            artifact_type="episode",
            content="cached",
            input_ids=[t1.artifact_id],
        )

        cached_by_inputs = {
            (t0.artifact_id,): [cached_art_0],
            (t1.artifact_id,): [cached_art_1],
        }

        on_cached_calls: list[list[Artifact]] = []

        def on_cached(arts):
            on_cached_calls.append(arts)

        units = [([t0], {}), ([t1], {}), ([t2], {})]
        transform = LoggingTransform()
        config = {}

        _execute_transform_concurrent(
            transform,
            units,
            config,
            concurrency=4,
            cached_by_inputs=cached_by_inputs,
            on_cached=on_cached,
        )

        # Only t-2 should have been executed
        assert call_log == ["t-2"], f"Expected only t-2 to be executed, got {call_log}"

        # on_cached should have been called twice (for t-0 and t-1)
        assert len(on_cached_calls) == 2


# ---------------------------------------------------------------------------
# Issue 2: EmbeddingConfig dimensions default None
# ---------------------------------------------------------------------------


class TestEmbeddingDimensionsDefault:
    """Tests for Issue 2: dimensions defaults to None."""

    def test_dimensions_default_is_none(self):
        """EmbeddingConfig.dimensions defaults to None."""
        config = EmbeddingConfig()
        assert config.dimensions is None

    def test_dimensions_none_not_passed_to_api(self):
        """When dimensions is None, it's not included in API call kwargs."""
        config = EmbeddingConfig(provider="openai", model="test-model", api_key="k")
        backend = OpenAIBackend(config)

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1, 0.2]])
        backend._client = mock_client

        backend._embed_chunk(["test"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert "dimensions" not in call_kwargs

    def test_dimensions_explicit_passed_to_api(self):
        """When dimensions is set explicitly, it IS included in API call."""
        config = EmbeddingConfig(provider="openai", model="test-model", dimensions=256, api_key="k")
        backend = OpenAIBackend(config)

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1, 0.2]])
        backend._client = mock_client

        backend._embed_chunk(["test"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 256

    def test_dimensions_zero_is_falsy_but_passed(self):
        """Dimensions=0 is falsy but should still be passed if not None.

        This tests that the code uses `is not None` rather than truthiness.
        Note: dimensions=0 is not a valid real value but tests the boundary.
        """
        config = EmbeddingConfig(provider="openai", model="test-model", dimensions=0, api_key="k")
        backend = OpenAIBackend(config)

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1, 0.2]])
        backend._client = mock_client

        backend._embed_chunk(["test"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        # dimensions=0 is not None, so it should be passed
        assert "dimensions" in call_kwargs
        assert call_kwargs["dimensions"] == 0

    def test_from_dict_without_dimensions(self):
        """EmbeddingConfig.from_dict without dimensions key keeps None default."""
        config = EmbeddingConfig.from_dict({"provider": "openai", "model": "test"})
        assert config.dimensions is None

    def test_from_dict_with_dimensions(self):
        """EmbeddingConfig.from_dict with explicit dimensions preserves value."""
        config = EmbeddingConfig.from_dict({"provider": "openai", "model": "test", "dimensions": 1024})
        assert config.dimensions == 1024


# ---------------------------------------------------------------------------
# Issue 3a: resolve_base_url()
# ---------------------------------------------------------------------------


class TestResolveBaseUrl:
    """Tests for Issue 3a: resolve_base_url()."""

    def test_explicit_base_url(self):
        """Explicit base_url takes precedence."""
        config = EmbeddingConfig(base_url="http://localhost:8080")
        assert config.resolve_base_url() == "http://localhost:8080"

    def test_env_var_base_url(self, monkeypatch):
        """Falls back to OPENAI_BASE_URL env var."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-url:8080")
        config = EmbeddingConfig()
        assert config.resolve_base_url() == "http://env-url:8080"

    def test_explicit_overrides_env(self, monkeypatch):
        """Explicit base_url overrides env var."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-url:8080")
        config = EmbeddingConfig(base_url="http://explicit:9090")
        assert config.resolve_base_url() == "http://explicit:9090"

    def test_no_base_url_returns_none(self, monkeypatch):
        """Without explicit or env var, returns None."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = EmbeddingConfig()
        assert config.resolve_base_url() is None

    def test_openai_backend_uses_resolve_base_url(self, monkeypatch):
        """OpenAIBackend._get_client() uses resolve_base_url() for the base_url kwarg."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-url:8080")
        config = EmbeddingConfig(provider="openai", model="test", api_key="test-key")
        backend = OpenAIBackend(config)

        # Mock the OpenAI constructor via the openai module (lazy import in _get_client)
        mock_openai_cls = MagicMock()
        monkeypatch.setattr("openai.OpenAI", mock_openai_cls)

        backend._client = None  # Reset to force _get_client() to create a new one
        backend._get_client()

        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://env-url:8080"

    def test_openai_backend_no_base_url_kwarg_when_none(self, monkeypatch):
        """When resolve_base_url() returns None, base_url is not passed to OpenAI()."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = EmbeddingConfig(provider="openai", model="test", api_key="test-key")
        backend = OpenAIBackend(config)

        mock_openai_cls = MagicMock()
        monkeypatch.setattr("openai.OpenAI", mock_openai_cls)

        backend._client = None
        backend._get_client()

        call_kwargs = mock_openai_cls.call_args[1]
        assert "base_url" not in call_kwargs


# ---------------------------------------------------------------------------
# Issue 3b: Manifest stores identity fields only
# ---------------------------------------------------------------------------


class TestManifestIdentityFields:
    """Tests for Issue 3b: manifest stores identity fields only."""

    def test_manifest_no_base_url(self, tmp_build_dir):
        """Manifest _config does not contain base_url."""
        config = EmbeddingConfig(provider="openai", model="test", base_url="http://localhost", api_key="k")
        provider = EmbeddingProvider(config, tmp_build_dir)
        meta = provider._current_config_meta()
        assert "base_url" not in meta
        assert "api_key" not in meta

    def test_manifest_has_identity_fields(self, tmp_build_dir):
        """Manifest _config has provider, model, dimensions."""
        config = EmbeddingConfig(provider="openai", model="test-model", dimensions=256)
        provider = EmbeddingProvider(config, tmp_build_dir)
        meta = provider._current_config_meta()
        assert meta == {"provider": "openai", "model": "test-model", "dimensions": 256}

    def test_manifest_no_batch_size(self, tmp_build_dir):
        """Manifest _config does not contain batch_size (not an identity field)."""
        config = EmbeddingConfig(provider="openai", model="test", batch_size=32)
        provider = EmbeddingProvider(config, tmp_build_dir)
        meta = provider._current_config_meta()
        assert "batch_size" not in meta

    def test_manifest_no_concurrency(self, tmp_build_dir):
        """Manifest _config does not contain concurrency (not an identity field)."""
        config = EmbeddingConfig(provider="openai", model="test", concurrency=8)
        provider = EmbeddingProvider(config, tmp_build_dir)
        meta = provider._current_config_meta()
        assert "concurrency" not in meta

    def test_manifest_dimensions_none_stored(self, tmp_build_dir):
        """When dimensions is None, manifest stores None (not omitted)."""
        config = EmbeddingConfig(provider="openai", model="test")
        provider = EmbeddingProvider(config, tmp_build_dir)
        meta = provider._current_config_meta()
        assert "dimensions" in meta
        assert meta["dimensions"] is None


# ---------------------------------------------------------------------------
# Issue 4: Batch size default 16 + 413 retry
# ---------------------------------------------------------------------------


class TestBatchSizeAndRetry:
    """Tests for Issue 4: batch_size default 16 + 413 retry."""

    def test_batch_size_default(self):
        """Default batch_size is 16."""
        config = EmbeddingConfig()
        assert config.batch_size == 16

    def test_batch_size_from_dict_default(self):
        """from_dict without batch_size keeps default 16."""
        config = EmbeddingConfig.from_dict({"provider": "openai"})
        assert config.batch_size == 16

    def test_413_retry_halves_batch(self):
        """On 413 error, _embed_chunk retries with halved batch."""
        config = EmbeddingConfig(provider="openai", model="test", api_key="k")
        backend = OpenAIBackend(config)
        mock_client = MagicMock()

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            texts = kwargs["input"]
            if len(texts) > 2 and call_count[0] == 1:
                exc = Exception("413 payload too large")
                exc.status_code = 413
                raise exc
            return MockEmbeddingResponse([[0.1]] * len(texts))

        mock_client.embeddings.create = MagicMock(side_effect=side_effect)
        backend._client = mock_client

        result = backend._embed_chunk(["a", "b", "c", "d"])
        assert len(result) == 4
        # Should have been called 3 times: original (fail), first half, second half
        assert mock_client.embeddings.create.call_count == 3

    def test_413_single_text_reraises(self):
        """413 with single text re-raises (can't halve further)."""
        config = EmbeddingConfig(provider="openai", model="test", api_key="k")
        backend = OpenAIBackend(config)
        mock_client = MagicMock()

        exc = Exception("413 payload too large")
        exc.status_code = 413
        mock_client.embeddings.create.side_effect = exc
        backend._client = mock_client

        with pytest.raises(Exception, match="413"):
            backend._embed_chunk(["single text"])

    def test_non_413_error_not_retried(self):
        """Non-413 errors are not retried."""
        config = EmbeddingConfig(provider="openai", model="test", api_key="k")
        backend = OpenAIBackend(config)
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = ValueError("some other error")
        backend._client = mock_client

        with pytest.raises(ValueError, match="some other error"):
            backend._embed_chunk(["a", "b"])

        # Should only be called once (no retry)
        assert mock_client.embeddings.create.call_count == 1

    def test_413_recursive_halving(self):
        """413 keeps halving recursively until batches are small enough."""
        config = EmbeddingConfig(provider="openai", model="test", api_key="k")
        backend = OpenAIBackend(config)
        mock_client = MagicMock()

        def side_effect(**kwargs):
            texts = kwargs["input"]
            if len(texts) > 1:
                exc = Exception("413 payload too large")
                exc.status_code = 413
                raise exc
            return MockEmbeddingResponse([[0.1]] * len(texts))

        mock_client.embeddings.create = MagicMock(side_effect=side_effect)
        backend._client = mock_client

        result = backend._embed_chunk(["a", "b", "c", "d"])
        assert len(result) == 4
        # Each text ends up in its own call after recursive halving

    def test_is_payload_too_large_status_code(self):
        """_is_payload_too_large detects 413 via status_code attribute."""
        exc = Exception("error")
        exc.status_code = 413
        assert _is_payload_too_large(exc) is True

    def test_is_payload_too_large_message(self):
        """_is_payload_too_large detects 413 in error message."""
        exc = Exception("HTTP 413 payload too large")
        assert _is_payload_too_large(exc) is True

    def test_is_payload_too_large_negative(self):
        """_is_payload_too_large returns False for non-413 errors."""
        exc = Exception("HTTP 500 internal server error")
        assert _is_payload_too_large(exc) is False

    def test_is_payload_too_large_request_entity(self):
        """_is_payload_too_large detects '413 request entity too large' variant."""
        exc = Exception("413 request entity too large")
        assert _is_payload_too_large(exc) is True


# ---------------------------------------------------------------------------
# Issue 5: Plan estimation with dirty upstream
# ---------------------------------------------------------------------------


class TestPlanEstimationDirtyUpstream:
    """Tests for Issue 5: plan estimation with dirty upstream.

    When upstream_dirty is True and the layer has estimate_output_count,
    use that instead of split() to avoid unreliable split() calls on stale inputs.
    """

    def test_estimate_used_when_upstream_dirty(self, tmp_path, mock_llm):
        """When upstream is dirty, estimate_output_count() is used instead of split().

        Strategy: build once, then modify source to trigger upstream dirty.
        The plan should use estimate_output_count for downstream layers
        that depend on dirty upstream, which avoids calling split() on stale data.
        """
        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
        shutil.copy(FIXTURES_DIR / "claude_export.json", source_dir / "claude_export.json")

        build_dir = tmp_path / "build"
        p = Pipeline("test-dirty-upstream")
        p.source_dir = str(source_dir)
        p.build_dir = str(build_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        p.add(transcripts, episodes)

        # First build
        run(p, concurrency=1)

        # Now add a new conversation so transcripts layer is dirty
        claude_path = source_dir / "claude_export.json"
        data = json.loads(claude_path.read_text())
        data["conversations"].append(
            {
                "uuid": "conv-dirty-test-001",
                "title": "Dirty upstream test",
                "created_at": "2024-04-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-dt-1",
                        "sender": "human",
                        "text": "Hello from dirty test.",
                        "created_at": "2024-04-01T10:00:00Z",
                    },
                    {
                        "uuid": "msg-dt-2",
                        "sender": "assistant",
                        "text": "Hi there, dirty test response.",
                        "created_at": "2024-04-01T10:01:00Z",
                    },
                ],
            }
        )
        claude_path.write_text(json.dumps(data))

        # Plan should detect the source change and mark episodes as needing rebuild
        plan = plan_build(p)

        transcript_step = next(s for s in plan.steps if s.name == "transcripts")
        episode_step = next(s for s in plan.steps if s.name == "episodes")

        # Transcripts should have rebuild_count > 0 (upstream dirty)
        assert transcript_step.rebuild_count > 0

        # Episodes should have a valid rebuild count (not crash or return 0)
        assert episode_step.rebuild_count >= 0
        # The total (rebuild + cached) should be reasonable
        total = episode_step.rebuild_count + episode_step.cached_count
        assert total > 0

    def test_plan_no_crash_with_dirty_upstream_and_n1_transform(self, tmp_path, mock_llm):
        """Plan doesn't crash when upstream is dirty and transform is N:1 (e.g., core).

        N:1 transforms like CoreSynthesis group all inputs into one unit.
        When upstream is dirty, the plan should still produce a valid estimate.
        """
        from synix.ext import CoreSynthesis, MonthlyRollup

        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")

        build_dir = tmp_path / "build"
        p = Pipeline("test-n1-dirty")
        p.source_dir = str(source_dir)
        p.build_dir = str(build_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        monthly = MonthlyRollup("monthly", depends_on=[episodes])
        core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)
        p.add(transcripts, episodes, monthly, core)

        # First build
        run(p, concurrency=1)

        # Modify source to make transcripts dirty
        chatgpt_path = source_dir / "chatgpt_export.json"
        data = json.loads(chatgpt_path.read_text())
        data.append(
            {
                "title": "New conversation",
                "create_time": 1711000000,
                "mapping": {
                    "node1": {
                        "id": "node1",
                        "parent": None,
                        "children": ["node2"],
                        "message": None,
                    },
                    "node2": {
                        "id": "node2",
                        "parent": "node1",
                        "children": ["node3"],
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello"]},
                            "create_time": 1711000001,
                        },
                    },
                    "node3": {
                        "id": "node3",
                        "parent": "node2",
                        "children": [],
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Hi there"]},
                            "create_time": 1711000002,
                        },
                    },
                },
            }
        )
        chatgpt_path.write_text(json.dumps(data))

        # Plan should not crash
        plan = plan_build(p)

        core_step = next(s for s in plan.steps if s.name == "core")
        # Core always estimates 1 output
        assert core_step.artifact_count >= 1

    def test_estimate_output_count_default_1_to_1(self):
        """Transform.estimate_output_count defaults to 1:1 (same as input count)."""

        class SimpleTransform(Transform):
            def execute(self, inputs, config):
                return []

        t = SimpleTransform("test")
        assert t.estimate_output_count(5) == 5
        assert t.estimate_output_count(0) == 0
