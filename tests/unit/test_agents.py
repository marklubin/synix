"""Tests for the Agent protocol and SynixLLMAgent implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synix.agents import Agent, Group, SynixLLMAgent
from synix.core.models import Artifact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal Agent protocol implementation for testing."""

    @property
    def agent_id(self):
        return "fake"

    def fingerprint_value(self):
        return "fp"

    def map(self, artifact, task_prompt):
        return "mapped"

    def reduce(self, artifacts, task_prompt):
        return "reduced"

    def group(self, artifacts, task_prompt):
        return [Group(key="g", artifacts=artifacts, content="grouped")]

    def fold(self, accumulated, artifact, step, total, task_prompt):
        return "folded"


def _make_artifact(label: str = "test", content: str = "hello") -> Artifact:
    return Artifact(label=label, artifact_type="test", content=content)


# ---------------------------------------------------------------------------
# TestGroup
# ---------------------------------------------------------------------------


class TestGroup:
    def test_construction(self):
        arts = [_make_artifact()]
        g = Group(key="topic-a", artifacts=arts, content="synthesized")
        assert g.key == "topic-a"
        assert g.artifacts == arts
        assert g.content == "synthesized"

    def test_empty_artifacts(self):
        g = Group(key="empty", artifacts=[], content="nothing")
        assert g.artifacts == []


# ---------------------------------------------------------------------------
# TestAgentProtocol
# ---------------------------------------------------------------------------


class TestAgentProtocol:
    def test_fake_agent_satisfies_protocol(self):
        agent = FakeAgent()
        assert isinstance(agent, Agent)

    def test_fake_agent_methods(self):
        agent = FakeAgent()
        art = _make_artifact()
        assert agent.agent_id == "fake"
        assert agent.fingerprint_value() == "fp"
        assert agent.map(art, "task") == "mapped"
        assert agent.reduce([art], "task") == "reduced"
        assert agent.fold("acc", art, 1, 5, "task") == "folded"
        groups = agent.group([art], "task")
        assert len(groups) == 1
        assert groups[0].key == "g"

    def test_missing_map_fails_isinstance(self):
        class NoMap:
            @property
            def agent_id(self):
                return "x"

            def fingerprint_value(self):
                return "x"

            def reduce(self, artifacts, task_prompt):
                return ""

            def group(self, artifacts, task_prompt):
                return []

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return ""

        assert not isinstance(NoMap(), Agent)

    def test_missing_fingerprint_fails_isinstance(self):
        class NoFingerprint:
            @property
            def agent_id(self):
                return "x"

            def map(self, artifact, task_prompt):
                return ""

            def reduce(self, artifacts, task_prompt):
                return ""

            def group(self, artifacts, task_prompt):
                return []

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return ""

        assert not isinstance(NoFingerprint(), Agent)

    def test_missing_reduce_fails_isinstance(self):
        class NoReduce:
            @property
            def agent_id(self):
                return "x"

            def fingerprint_value(self):
                return "x"

            def map(self, artifact, task_prompt):
                return ""

            def group(self, artifacts, task_prompt):
                return []

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return ""

        assert not isinstance(NoReduce(), Agent)

    def test_missing_group_fails_isinstance(self):
        class NoGroup:
            @property
            def agent_id(self):
                return "x"

            def fingerprint_value(self):
                return "x"

            def map(self, artifact, task_prompt):
                return ""

            def reduce(self, artifacts, task_prompt):
                return ""

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return ""

        assert not isinstance(NoGroup(), Agent)

    def test_missing_fold_fails_isinstance(self):
        class NoFold:
            @property
            def agent_id(self):
                return "x"

            def fingerprint_value(self):
                return "x"

            def map(self, artifact, task_prompt):
                return ""

            def reduce(self, artifacts, task_prompt):
                return ""

            def group(self, artifacts, task_prompt):
                return []

        assert not isinstance(NoFold(), Agent)


# ---------------------------------------------------------------------------
# TestSynixLLMAgent
# ---------------------------------------------------------------------------


class TestSynixLLMAgent:
    def test_creation(self):
        agent = SynixLLMAgent(name="summarizer", prompt_key="summarize")
        assert agent.name == "summarizer"
        assert agent.prompt_key == "summarize"
        assert agent.llm_config is None
        assert agent.description == ""

    def test_agent_id(self):
        agent = SynixLLMAgent(name="my-agent", prompt_key="key")
        assert agent.agent_id == "my-agent"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must have a name"):
            SynixLLMAgent(name="", prompt_key="key")

    def test_empty_prompt_key_raises(self):
        with pytest.raises(ValueError, match="must have a prompt_key"):
            SynixLLMAgent(name="agent", prompt_key="")

    def test_with_llm_config(self):
        agent = SynixLLMAgent(
            name="agent",
            prompt_key="key",
            llm_config={"model": "claude-sonnet-4-20250514", "temperature": 0.5},
        )
        assert agent.llm_config["model"] == "claude-sonnet-4-20250514"

    def test_with_description(self):
        agent = SynixLLMAgent(
            name="agent", prompt_key="key", description="Does stuff"
        )
        assert agent.description == "Does stuff"


# ---------------------------------------------------------------------------
# TestSynixLLMAgentPromptStore
# ---------------------------------------------------------------------------


class TestSynixLLMAgentPromptStore:
    def test_bind_prompt_store(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("my-prompt", "You are a helpful agent.")
        agent = SynixLLMAgent(name="agent", prompt_key="my-prompt")
        result = agent.bind_prompt_store(store)
        assert result is agent  # returns self for chaining
        assert agent.instructions == "You are a helpful agent."
        store.close()

    def test_instructions_without_store_raises(self):
        agent = SynixLLMAgent(name="agent", prompt_key="key")
        with pytest.raises(ValueError, match="no prompt store"):
            _ = agent.instructions

    def test_instructions_key_not_found_raises(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        agent = SynixLLMAgent(name="agent", prompt_key="missing-key")
        agent.bind_prompt_store(store)
        with pytest.raises(ValueError, match="not found in store"):
            _ = agent.instructions
        store.close()

    def test_instructions_picks_up_edits(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("prompt", "Version 1")
        agent = SynixLLMAgent(name="agent", prompt_key="prompt")
        agent.bind_prompt_store(store)
        assert agent.instructions == "Version 1"

        store.put("prompt", "Version 2")
        assert agent.instructions == "Version 2"
        store.close()

    def test_fingerprint_changes_with_prompt_edit(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("prompt", "Original instructions")
        agent = SynixLLMAgent(name="agent", prompt_key="prompt")
        agent.bind_prompt_store(store)

        fp1 = agent.fingerprint_value()
        store.put("prompt", "Updated instructions")
        fp2 = agent.fingerprint_value()
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# TestSynixLLMAgentMap
# ---------------------------------------------------------------------------


class TestSynixLLMAgentMap:
    def test_map_renders_and_calls_llm(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("map-prompt", "You are a summarizer.")
        agent = SynixLLMAgent(name="mapper", prompt_key="map-prompt")
        agent.bind_prompt_store(store)

        mock_response = MagicMock()
        mock_response.content = "Summary output"

        with patch("synix.build.llm_client.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = mock_response

            result = agent.map(_make_artifact(content="raw text"), task_prompt="Summarize: raw text")

        assert result == "Summary output"
        # Verify LLM was called with system + user messages
        call_args = instance.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # task_prompt is the user message now
        assert "Summarize: raw text" in messages[1]["content"]
        store.close()


# ---------------------------------------------------------------------------
# TestSynixLLMAgentReduce
# ---------------------------------------------------------------------------


class TestSynixLLMAgentReduce:
    def test_reduce_passes_task_prompt_to_llm(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("reduce-prompt", "You are a reducer agent.")
        agent = SynixLLMAgent(name="reducer", prompt_key="reduce-prompt")
        agent.bind_prompt_store(store)

        arts = [
            _make_artifact(label="a", content="first"),
            _make_artifact(label="b", content="second"),
        ]

        mock_response = MagicMock()
        mock_response.content = "Combined output"

        with patch("synix.build.llm_client.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = mock_response

            task_prompt = "Combine these 2 items:\nfirst\nsecond"
            result = agent.reduce(arts, task_prompt)

        assert result == "Combined output"
        call_args = instance.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        # System message is agent instructions, user message is task_prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a reducer agent."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == task_prompt
        store.close()


# ---------------------------------------------------------------------------
# TestSynixLLMAgentFold
# ---------------------------------------------------------------------------


class TestSynixLLMAgentFold:
    def test_fold_passes_task_prompt_to_llm(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("fold-prompt", "You are a fold agent.")
        agent = SynixLLMAgent(name="folder", prompt_key="fold-prompt")
        agent.bind_prompt_store(store)

        mock_response = MagicMock()
        mock_response.content = "Folded result"

        with patch("synix.build.llm_client.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = mock_response

            task_prompt = "Step 3/10. So far: previous state. New: new item"
            result = agent.fold(
                accumulated="previous state",
                artifact=_make_artifact(content="new item"),
                step=3,
                total=10,
                task_prompt=task_prompt,
            )

        assert result == "Folded result"
        call_args = instance.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        # System message is agent instructions, user message is the task_prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a fold agent."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == task_prompt
        store.close()


# ---------------------------------------------------------------------------
# TestSynixLLMAgentGroup
# ---------------------------------------------------------------------------


class TestSynixLLMAgentGroup:
    def test_group_raises_not_implemented(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("group-prompt", "Group these")
        agent = SynixLLMAgent(name="grouper", prompt_key="group-prompt")
        agent.bind_prompt_store(store)

        with pytest.raises(NotImplementedError, match="does not implement group"):
            agent.group([_make_artifact()], "Group these artifacts")
        store.close()


# ---------------------------------------------------------------------------
# TestSynixLLMAgentFromFile
# ---------------------------------------------------------------------------


class TestSynixLLMAgentFromFile:
    def test_from_file_seeds_prompt_store(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        # Write instructions file
        instructions_file = tmp_path / "instructions.txt"
        instructions_file.write_text("You are a specialized agent.")

        store = PromptStore(tmp_path / "prompts.db")
        agent = SynixLLMAgent.from_file(
            name="file-agent",
            prompt_key="file-key",
            instructions_path=instructions_file,
            prompt_store=store,
        )

        assert agent.agent_id == "file-agent"
        assert agent.instructions == "You are a specialized agent."
        # Verify it was actually stored
        assert store.get("file-key") == "You are a specialized agent."
        store.close()

    def test_from_file_with_extra_kwargs(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        instructions_file = tmp_path / "instr.txt"
        instructions_file.write_text("Do things.")

        store = PromptStore(tmp_path / "prompts.db")
        agent = SynixLLMAgent.from_file(
            name="custom",
            prompt_key="custom-key",
            instructions_path=instructions_file,
            prompt_store=store,
            llm_config={"model": "gpt-4"},
            description="Custom agent",
        )

        assert agent.llm_config == {"model": "gpt-4"}
        assert agent.description == "Custom agent"
        store.close()


# ---------------------------------------------------------------------------
# TestFingerprintValue
# ---------------------------------------------------------------------------


class TestFingerprintValue:
    def test_deterministic(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("key", "instructions")
        agent = SynixLLMAgent(name="agent", prompt_key="key")
        agent.bind_prompt_store(store)

        fp1 = agent.fingerprint_value()
        fp2 = agent.fingerprint_value()
        assert fp1 == fp2
        store.close()

    def test_changes_with_llm_config(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("key", "instructions")

        agent1 = SynixLLMAgent(name="agent", prompt_key="key")
        agent1.bind_prompt_store(store)

        agent2 = SynixLLMAgent(
            name="agent", prompt_key="key", llm_config={"model": "gpt-4"}
        )
        agent2.bind_prompt_store(store)

        assert agent1.fingerprint_value() != agent2.fingerprint_value()
        store.close()

    def test_changes_with_prompt_content(self, tmp_path):
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("key", "version A")
        agent = SynixLLMAgent(name="agent", prompt_key="key")
        agent.bind_prompt_store(store)
        fp_a = agent.fingerprint_value()

        store.put("key", "version B")
        fp_b = agent.fingerprint_value()

        assert fp_a != fp_b
        store.close()

    def test_without_store_raises(self):
        """Without a bound store, fingerprint_value() raises for cache safety."""
        agent = SynixLLMAgent(name="agent", prompt_key="key")
        with pytest.raises(ValueError, match="no prompt store"):
            agent.fingerprint_value()

    def test_same_config_different_names_same_fingerprint(self, tmp_path):
        """Fingerprint is based on content/config, not agent name."""
        from synix.server.prompt_store import PromptStore

        store = PromptStore(tmp_path / "test.db")
        store.put("shared-key", "same instructions")

        agent1 = SynixLLMAgent(name="agent-1", prompt_key="shared-key")
        agent1.bind_prompt_store(store)

        agent2 = SynixLLMAgent(name="agent-2", prompt_key="shared-key")
        agent2.bind_prompt_store(store)

        assert agent1.fingerprint_value() == agent2.fingerprint_value()
        store.close()
