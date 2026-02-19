"""Tests for issues #6, #7, #8 fixes.

Issue 6: Ext transforms don't propagate input metadata to output artifacts
Issue 7: openai-compatible and openai providers both resolve to OPENAI_API_KEY
Issue 8: No built-in adapter for Claude Code .jsonl session transcripts
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from synix import Artifact
from synix.core.config import EmbeddingConfig, LLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(label: str, metadata: dict | None = None) -> Artifact:
    """Create an artifact with optional metadata for testing."""
    return Artifact(
        label=label,
        artifact_type="transcript",
        content=f"Content of {label}",
        metadata=metadata or {},
    )


class _MockContentBlock:
    def __init__(self, text: str):
        self.text = text


class _MockResponse:
    def __init__(self, text: str = "Mock LLM output"):
        self.content = [_MockContentBlock(text)]
        self.model = "test-model"
        self.usage = MagicMock(input_tokens=10, output_tokens=20)


def _mock_llm_config(monkeypatch):
    """Mock the LLM client for ext transforms."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _MockResponse()

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
    return mock_client


def _base_config():
    """Return a minimal config dict for ext transforms."""
    return {
        "llm_config": {"provider": "anthropic", "model": "test", "max_tokens": 100},
    }


# ---------------------------------------------------------------------------
# Issue 6: Metadata propagation in ext transforms
# ---------------------------------------------------------------------------


class TestMapSynthesisMetadata:
    """Tests for Issue 6: MapSynthesis metadata propagation."""

    def test_propagates_input_metadata(self, monkeypatch):
        """MapSynthesis auto-propagates input metadata to output."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import MapSynthesis

        t = MapSynthesis("test-map", prompt="Summarize: {artifact}", artifact_type="summary")
        inp = _make_artifact("inp-1", metadata={"project": "foo", "date": "2025-01-01"})
        results = t.execute([inp], _base_config())

        assert len(results) == 1
        out = results[0]
        assert out.metadata["project"] == "foo"
        assert out.metadata["date"] == "2025-01-01"

    def test_metadata_fn_overrides(self, monkeypatch):
        """metadata_fn output overrides input metadata."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import MapSynthesis

        t = MapSynthesis(
            "test-map",
            prompt="Summarize: {artifact}",
            artifact_type="summary",
            metadata_fn=lambda inp: {"project": "overridden", "custom": "value"},
        )
        inp = _make_artifact("inp-1", metadata={"project": "foo", "date": "2025-01-01"})
        results = t.execute([inp], _base_config())

        out = results[0]
        assert out.metadata["project"] == "overridden"
        assert out.metadata["custom"] == "value"
        # Input metadata still present where not overridden
        assert out.metadata["date"] == "2025-01-01"

    def test_source_label_always_set(self, monkeypatch):
        """source_label is always set in output metadata."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import MapSynthesis

        t = MapSynthesis("test-map", prompt="Summarize: {artifact}", artifact_type="summary")
        inp = _make_artifact("my-input")
        results = t.execute([inp], _base_config())

        assert results[0].metadata["source_label"] == "my-input"

    def test_source_label_overrides_input_source_label(self, monkeypatch):
        """Transform's source_label takes precedence over any input source_label."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import MapSynthesis

        t = MapSynthesis("test-map", prompt="Summarize: {artifact}", artifact_type="summary")
        inp = _make_artifact("my-input", metadata={"source_label": "old-label"})
        results = t.execute([inp], _base_config())

        # source_label should be the actual input label, not the old propagated one
        assert results[0].metadata["source_label"] == "my-input"


class TestGroupSynthesisMetadata:
    """Tests for Issue 6: GroupSynthesis metadata_fn."""

    def test_metadata_fn_adds_custom_fields(self, monkeypatch):
        """metadata_fn(group_key, inputs) adds custom fields to output."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import GroupSynthesis

        t = GroupSynthesis(
            "test-group",
            group_by="project",
            prompt="Summarize group '{group_key}': {artifacts}",
            artifact_type="summary",
            metadata_fn=lambda gk, inputs: {"custom_count": len(inputs), "region": "us-east"},
        )
        inp1 = _make_artifact("inp-1", metadata={"project": "alpha"})
        inp2 = _make_artifact("inp-2", metadata={"project": "alpha"})

        results = t.execute([inp1, inp2], _base_config())
        assert len(results) == 1
        out = results[0]
        assert out.metadata["group_key"] == "alpha"
        assert out.metadata["input_count"] == 2
        assert out.metadata["custom_count"] == 2
        assert out.metadata["region"] == "us-east"

    def test_without_metadata_fn(self, monkeypatch):
        """Without metadata_fn, only default fields are set."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import GroupSynthesis

        t = GroupSynthesis(
            "test-group",
            group_by="project",
            prompt="Summarize: {artifacts}",
            artifact_type="summary",
        )
        inp = _make_artifact("inp-1", metadata={"project": "alpha"})
        results = t.execute([inp], _base_config())

        out = results[0]
        assert out.metadata == {"group_key": "alpha", "input_count": 1}


class TestReduceSynthesisMetadata:
    """Tests for Issue 6: ReduceSynthesis metadata_fn."""

    def test_metadata_fn_adds_custom_fields(self, monkeypatch):
        """metadata_fn(inputs) adds custom fields to output."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import ReduceSynthesis

        t = ReduceSynthesis(
            "test-reduce",
            prompt="Combine: {artifacts}",
            label="reduced",
            artifact_type="summary",
            metadata_fn=lambda inputs: {"sources": [a.label for a in inputs]},
        )
        inp1 = _make_artifact("inp-1")
        inp2 = _make_artifact("inp-2")
        results = t.execute([inp1, inp2], _base_config())

        assert len(results) == 1
        out = results[0]
        assert out.metadata["input_count"] == 2
        assert set(out.metadata["sources"]) == {"inp-1", "inp-2"}


class TestFoldSynthesisMetadata:
    """Tests for Issue 6: FoldSynthesis metadata_fn."""

    def test_metadata_fn_adds_custom_fields(self, monkeypatch):
        """metadata_fn(inputs) adds custom fields to output."""
        _mock_llm_config(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from synix.ext import FoldSynthesis

        t = FoldSynthesis(
            "test-fold",
            prompt="Update: {accumulated}\nNew: {artifact}",
            initial="Start",
            label="folded",
            artifact_type="summary",
            metadata_fn=lambda inputs: {"total_steps": len(inputs)},
        )
        inp1 = _make_artifact("inp-1")
        inp2 = _make_artifact("inp-2")
        results = t.execute([inp1, inp2], _base_config())

        assert len(results) == 1
        out = results[0]
        assert out.metadata["input_count"] == 2
        assert out.metadata["total_steps"] == 2


class TestMetadataFnFingerprint:
    """Tests for Issue 6: metadata_fn affects fingerprint."""

    def test_metadata_fn_changes_fingerprint(self):
        """Changing metadata_fn produces a different fingerprint."""
        from synix.ext import MapSynthesis

        fn_a = lambda inp: {"version": "a"}  # noqa: E731
        fn_b = lambda inp: {"version": "b"}  # noqa: E731

        t_a = MapSynthesis("test", prompt="p", artifact_type="s", metadata_fn=fn_a)
        t_b = MapSynthesis("test", prompt="p", artifact_type="s", metadata_fn=fn_b)
        t_none = MapSynthesis("test", prompt="p", artifact_type="s")

        fp_a = t_a.compute_fingerprint({})
        fp_b = t_b.compute_fingerprint({})
        fp_none = t_none.compute_fingerprint({})

        assert fp_a.digest != fp_b.digest
        assert fp_a.digest != fp_none.digest

    def test_metadata_fn_changes_cache_key(self):
        """Changing metadata_fn produces a different cache key."""
        from synix.ext import ReduceSynthesis

        fn_a = lambda inputs: {"v": "a"}  # noqa: E731
        fn_b = lambda inputs: {"v": "b"}  # noqa: E731

        t_a = ReduceSynthesis("test", prompt="p", label="l", metadata_fn=fn_a)
        t_b = ReduceSynthesis("test", prompt="p", label="l", metadata_fn=fn_b)
        t_none = ReduceSynthesis("test", prompt="p", label="l")

        key_a = t_a.get_cache_key({})
        key_b = t_b.get_cache_key({})
        key_none = t_none.get_cache_key({})

        assert key_a != key_b
        assert key_a != key_none


# ---------------------------------------------------------------------------
# Issue 7: API key env var collision
# ---------------------------------------------------------------------------


class TestLLMConfigApiKeyResolution:
    """Tests for Issue 7: LLMConfig API key resolution."""

    def test_api_key_env_field(self, monkeypatch):
        """api_key_env reads from the specified env var."""
        monkeypatch.setenv("TOGETHER_API_KEY", "together-secret")
        monkeypatch.delenv("SYNIX_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = LLMConfig(provider="openai-compatible", api_key_env="TOGETHER_API_KEY")
        assert config.resolve_api_key() == "together-secret"

    def test_synix_api_key_precedence(self, monkeypatch):
        """SYNIX_API_KEY takes precedence over provider-specific env vars."""
        monkeypatch.setenv("SYNIX_API_KEY", "synix-master")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = LLMConfig(provider="openai")
        assert config.resolve_api_key() == "synix-master"

    def test_explicit_api_key_wins(self, monkeypatch):
        """Explicit api_key beats everything."""
        monkeypatch.setenv("SYNIX_API_KEY", "synix-master")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        config = LLMConfig(provider="openai", api_key="explicit-key")
        assert config.resolve_api_key() == "explicit-key"

    def test_api_key_env_beats_synix_api_key(self, monkeypatch):
        """api_key_env takes precedence over SYNIX_API_KEY."""
        monkeypatch.setenv("TOGETHER_API_KEY", "together-secret")
        monkeypatch.setenv("SYNIX_API_KEY", "synix-master")

        config = LLMConfig(provider="openai-compatible", api_key_env="TOGETHER_API_KEY")
        assert config.resolve_api_key() == "together-secret"

    def test_provider_default_fallback(self, monkeypatch):
        """Without api_key_env or SYNIX_API_KEY, falls back to provider default."""
        monkeypatch.delenv("SYNIX_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        config = LLMConfig(provider="anthropic")
        assert config.resolve_api_key() == "ant-key"

    def test_api_key_env_from_dict(self):
        """from_dict() reads api_key_env."""
        config = LLMConfig.from_dict({"api_key_env": "MY_CUSTOM_KEY"})
        assert config.api_key_env == "MY_CUSTOM_KEY"

    def test_api_key_env_empty_var_falls_through(self, monkeypatch):
        """If api_key_env points to empty/unset var, falls through to next level."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        monkeypatch.setenv("SYNIX_API_KEY", "synix-fallback")

        config = LLMConfig(provider="openai", api_key_env="MISSING_KEY")
        assert config.resolve_api_key() == "synix-fallback"


class TestEmbeddingConfigApiKeyResolution:
    """Tests for Issue 7: EmbeddingConfig API key resolution."""

    def test_api_key_env_field(self, monkeypatch):
        """api_key_env reads from the specified env var."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-secret")
        monkeypatch.delenv("SYNIX_EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = EmbeddingConfig(provider="openai", api_key_env="VOYAGE_API_KEY")
        assert config.resolve_api_key() == "voyage-secret"

    def test_synix_embedding_api_key_precedence(self, monkeypatch):
        """SYNIX_EMBEDDING_API_KEY takes precedence over OPENAI_API_KEY."""
        monkeypatch.setenv("SYNIX_EMBEDDING_API_KEY", "synix-embed")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        config = EmbeddingConfig(provider="openai")
        assert config.resolve_api_key() == "synix-embed"

    def test_explicit_api_key_wins(self, monkeypatch):
        """Explicit api_key beats everything."""
        monkeypatch.setenv("SYNIX_EMBEDDING_API_KEY", "synix-embed")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        config = EmbeddingConfig(provider="openai", api_key="explicit-key")
        assert config.resolve_api_key() == "explicit-key"

    def test_api_key_env_from_dict(self):
        """from_dict() reads api_key_env."""
        config = EmbeddingConfig.from_dict({"api_key_env": "CUSTOM_EMBED_KEY"})
        assert config.api_key_env == "CUSTOM_EMBED_KEY"

    def test_fallback_to_openai_api_key(self, monkeypatch):
        """Without any overrides, falls back to OPENAI_API_KEY."""
        monkeypatch.delenv("SYNIX_EMBEDDING_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")

        config = EmbeddingConfig(provider="openai")
        assert config.resolve_api_key() == "openai-fallback"


# ---------------------------------------------------------------------------
# Issue 8: Claude Code JSONL adapter
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_claude_code_entry(
    role: str,
    content: str | list,
    *,
    timestamp: str = "2025-06-15T10:30:00Z",
    session_id: str = "abc123",
    slug: str = "test-session",
    cwd: str = "/home/user/project",
    git_branch: str = "main",
) -> dict:
    """Create a Claude Code JSONL entry."""
    return {
        "type": role,
        "message": {"role": role, "content": content},
        "timestamp": timestamp,
        "sessionId": session_id,
        "slug": slug,
        "cwd": cwd,
        "gitBranch": git_branch,
    }


class TestParseClaudeCodeBasic:
    """Tests for Issue 8: Claude Code JSONL parsing."""

    def test_basic_two_turn(self, tmp_path):
        """2-turn JSONL produces one Artifact with correct content."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            _make_claude_code_entry("user", "Hello, how are you?"),
            _make_claude_code_entry("assistant", "I'm doing well, thanks!"),
        ]
        path = tmp_path / "session1.jsonl"
        _write_jsonl(path, entries)

        results = parse_claude_code(path)
        assert len(results) == 1
        art = results[0]
        assert art.artifact_type == "transcript"
        assert "User: Hello, how are you?" in art.content
        assert "Assistant: I'm doing well, thanks!" in art.content
        assert art.label == "t-claude-code-session1"

    def test_content_blocks_array(self, tmp_path):
        """Content as array of {type: "text", text: ...} blocks is extracted."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            _make_claude_code_entry("user", "Question"),
            _make_claude_code_entry(
                "assistant",
                [
                    {"type": "text", "text": "First paragraph."},
                    {"type": "text", "text": "Second paragraph."},
                ],
            ),
        ]
        path = tmp_path / "session2.jsonl"
        _write_jsonl(path, entries)

        results = parse_claude_code(path)
        assert len(results) == 1
        assert "First paragraph." in results[0].content
        assert "Second paragraph." in results[0].content

    def test_skips_tool_blocks(self, tmp_path):
        """tool_use and tool_result blocks are filtered out."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            _make_claude_code_entry("user", "Run the test"),
            _make_claude_code_entry(
                "assistant",
                [
                    {"type": "text", "text": "Running tests now."},
                    {"type": "tool_use", "id": "tool1", "name": "bash", "input": {"cmd": "pytest"}},
                ],
            ),
            {"type": "tool_result", "tool_use_id": "tool1", "content": "All tests passed"},
            _make_claude_code_entry(
                "assistant",
                [
                    {"type": "text", "text": "Tests passed!"},
                    {"type": "thinking", "thinking": "Let me check..."},
                ],
            ),
        ]
        path = tmp_path / "session3.jsonl"
        _write_jsonl(path, entries)

        results = parse_claude_code(path)
        assert len(results) == 1
        content = results[0].content
        assert "Running tests now." in content
        assert "Tests passed!" in content
        assert "tool_use" not in content
        assert "tool_result" not in content
        assert "thinking" not in content.lower() or "Let me check" not in content

    def test_too_few_turns(self, tmp_path):
        """Sessions with < 2 meaningful turns return empty list."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            _make_claude_code_entry("user", "Hello"),
        ]
        path = tmp_path / "short.jsonl"
        _write_jsonl(path, entries)

        assert parse_claude_code(path) == []

    def test_metadata_extraction(self, tmp_path):
        """Metadata fields are correctly extracted."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            _make_claude_code_entry(
                "user",
                "Question",
                timestamp="2025-06-15T10:30:00Z",
                slug="my-session",
                cwd="/home/user/project",
                git_branch="feature-x",
            ),
            _make_claude_code_entry("assistant", "Answer"),
        ]
        path = tmp_path / "meta-test.jsonl"
        _write_jsonl(path, entries)

        results = parse_claude_code(path)
        meta = results[0].metadata
        assert meta["source"] == "claude-code"
        assert meta["session_id"] == "meta-test"
        assert meta["date"] == "2025-06-15"
        assert meta["cwd"] == "/home/user/project"
        assert meta["git_branch"] == "feature-x"
        assert meta["title"] == "my-session"
        assert meta["message_count"] == 2

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSON lines are skipped gracefully."""
        from synix.adapters.claude_code import parse_claude_code

        path = tmp_path / "malformed.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_make_claude_code_entry("user", "Hello")) + "\n")
            f.write("this is not json\n")
            f.write("{broken json\n")
            f.write(json.dumps(_make_claude_code_entry("assistant", "Hi there")) + "\n")

        results = parse_claude_code(path)
        assert len(results) == 1
        assert "User: Hello" in results[0].content
        assert "Assistant: Hi there" in results[0].content

    def test_non_claude_code_jsonl(self, tmp_path):
        """JSONL without Claude Code fields returns empty list."""
        from synix.adapters.claude_code import parse_claude_code

        entries = [
            {"event": "page_view", "url": "/home", "ts": 1234567890},
            {"event": "click", "target": "button", "ts": 1234567891},
        ]
        path = tmp_path / "analytics.jsonl"
        _write_jsonl(path, entries)

        assert parse_claude_code(path) == []

    def test_empty_file(self, tmp_path):
        """Empty JSONL file returns empty list."""
        from synix.adapters.claude_code import parse_claude_code

        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert parse_claude_code(path) == []


class TestRegistryJsonlExtension:
    """Tests for Issue 8: .jsonl registered in adapter registry."""

    def test_jsonl_maps_to_parse_claude_code(self):
        """The .jsonl extension is registered and maps to parse_claude_code."""
        from synix.adapters.claude_code import parse_claude_code
        from synix.adapters.registry import get_adapter

        adapter = get_adapter(Path("test.jsonl"))
        assert adapter is parse_claude_code

    def test_jsonl_in_supported_extensions(self):
        """The .jsonl extension appears in supported extensions."""
        from synix.adapters.registry import get_supported_extensions

        assert ".jsonl" in get_supported_extensions()


class TestMiddleTruncation:
    """Tests for Issue 8: middle truncation of long transcripts."""

    def test_long_transcript_truncated(self, tmp_path):
        """Transcripts exceeding max_chars are middle-truncated."""
        from synix.adapters.claude_code import parse_claude_code

        # Create entries with very long content
        long_text = "x" * 50_000
        entries = [
            _make_claude_code_entry("user", long_text),
            _make_claude_code_entry("assistant", long_text),
        ]
        path = tmp_path / "long.jsonl"
        _write_jsonl(path, entries)

        results = parse_claude_code(path, max_chars=1000)
        assert len(results) == 1
        assert len(results[0].content) <= 1100  # some slack for the marker
        assert "[... middle truncated ...]" in results[0].content
