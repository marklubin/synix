"""Tests for Agent gateway integration in generic synthesis transforms."""

from __future__ import annotations

import pytest

from synix import Artifact
from synix.agents import Agent, AgentRequest, AgentResult
from synix.transforms import FoldSynthesis, GroupSynthesis, MapSynthesis, ReduceSynthesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Test double satisfying the Agent protocol."""

    def __init__(self, response: str = "agent output", fingerprint: str = "test-agent-fp"):
        self._response = response
        self._fingerprint = fingerprint
        self.calls: list[AgentRequest] = []

    def write(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request)
        return AgentResult(content=self._response)

    def fingerprint_value(self) -> str:
        return self._fingerprint


def _make_artifact(label: str, content: str = "content", **metadata) -> Artifact:
    return Artifact(
        label=label,
        artifact_type="test",
        content=content,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# MapSynthesis + Agent
# ---------------------------------------------------------------------------


class TestMapWithAgent:
    def test_agent_write_called_with_rendered_prompt(self):
        agent = FakeAgent()
        t = MapSynthesis(
            "ws",
            prompt="Analyze: {artifact}",
            artifact_type="analysis",
            agent=agent,
        )
        inp = _make_artifact("bio-alice", "Alice is an engineer.")
        t.execute([inp], {"llm_config": {}})

        assert len(agent.calls) == 1
        req = agent.calls[0]
        assert "Alice is an engineer." in req.prompt
        assert req.metadata["transform_name"] == "ws"
        assert req.metadata["shape"] == "map"
        assert req.metadata["input_labels"] == ["bio-alice"]
        assert req.metadata["artifact_type"] == "analysis"

    def test_artifact_has_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="map-fp-123")
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].agent_fingerprint == "map-fp-123"

    def test_artifact_content_from_agent(self):
        agent = FakeAgent(response="agent-generated analysis")
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].content == "agent-generated analysis"

    def test_model_config_none_with_agent(self):
        agent = FakeAgent()
        t = MapSynthesis("ws", prompt="x", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {"model": "claude-sonnet-4-20250514"}})

        assert results[0].model_config is None

    def test_prompt_id_still_set(self):
        agent = FakeAgent()
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].prompt_id is not None
        assert results[0].prompt_id.startswith("map_synthesis_v")


# ---------------------------------------------------------------------------
# ReduceSynthesis + Agent
# ---------------------------------------------------------------------------


class TestReduceWithAgent:
    def test_agent_write_called(self):
        agent = FakeAgent()
        t = ReduceSynthesis(
            "team",
            prompt="Analyze: {artifacts}",
            label="team-dynamics",
            artifact_type="dynamics",
            agent=agent,
        )
        inputs = [_make_artifact(f"ws-{i}", f"profile {i}") for i in range(3)]
        results = t.execute(inputs, {"llm_config": {}})

        assert len(agent.calls) == 1
        req = agent.calls[0]
        assert req.metadata["shape"] == "reduce"
        assert req.metadata["count"] == 3
        assert len(req.metadata["input_labels"]) == 3
        assert results[0].agent_fingerprint == "test-agent-fp"
        assert results[0].model_config is None

    def test_artifact_content_from_agent(self):
        agent = FakeAgent(response="reduced output")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        inputs = [_make_artifact(f"a-{i}") for i in range(2)]
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].content == "reduced output"


# ---------------------------------------------------------------------------
# GroupSynthesis + Agent
# ---------------------------------------------------------------------------


class TestGroupWithAgent:
    def test_agent_called_per_group(self):
        agent = FakeAgent()
        inputs = [
            _make_artifact("ep-1", "content 1", team="alpha"),
            _make_artifact("ep-2", "content 2", team="beta"),
            _make_artifact("ep-3", "content 3", team="alpha"),
        ]
        t = GroupSynthesis(
            "team-summaries",
            group_by="team",
            prompt="Summarize team '{group_key}':\n\n{artifacts}",
            artifact_type="team_summary",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        # Two groups -> two agent calls
        assert len(agent.calls) == 2
        assert len(results) == 2

        # Check metadata on each call
        shapes = {call.metadata["group_key"] for call in agent.calls}
        assert shapes == {"alpha", "beta"}
        for call in agent.calls:
            assert call.metadata["shape"] == "group"
            assert call.metadata["transform_name"] == "team-summaries"

    def test_artifacts_have_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="group-fp")
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "s",
            group_by="team",
            prompt="{artifacts}",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint == "group-fp"
        assert results[0].model_config is None


# ---------------------------------------------------------------------------
# FoldSynthesis + Agent
# ---------------------------------------------------------------------------


class TestFoldWithAgent:
    def test_agent_called_per_step(self):
        agent = FakeAgent(response="accumulated")
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = FoldSynthesis(
            "progressive",
            prompt="Current: {accumulated}\nNew: {artifact}",
            initial="Empty.",
            label="progressive",
            artifact_type="progressive",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        # One call per input
        assert len(agent.calls) == 3
        assert len(results) == 1
        assert results[0].content == "accumulated"

        # Verify step metadata
        steps = [call.metadata["step"] for call in agent.calls]
        assert steps == [1, 2, 3]
        for call in agent.calls:
            assert call.metadata["shape"] == "fold"
            assert call.metadata["total"] == 3

    def test_artifact_has_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="fold-fp")
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis(
            "fold",
            prompt="{accumulated}\n{artifact}",
            label="out",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint == "fold-fp"
        assert results[0].model_config is None

    def test_fold_checkpoint_still_written(self):
        agent = FakeAgent(response="folded")
        inputs = [_make_artifact(f"ep-{i}") for i in range(2)]
        t = FoldSynthesis(
            "fold",
            prompt="{accumulated}\n{artifact}",
            label="out",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert "_fold_checkpoint" in results[0].metadata
        cp = results[0].metadata["_fold_checkpoint"]
        assert cp["version"] == 1
        assert len(cp["seen_inputs"]) == 2


# ---------------------------------------------------------------------------
# Prompt still required
# ---------------------------------------------------------------------------


class TestPromptStillRequired:
    def test_map_requires_prompt(self):
        agent = FakeAgent()
        with pytest.raises(TypeError):
            MapSynthesis("ws", agent=agent)

    def test_reduce_requires_prompt(self):
        agent = FakeAgent()
        with pytest.raises(TypeError):
            ReduceSynthesis("r", label="out", agent=agent)

    def test_group_requires_prompt(self):
        agent = FakeAgent()
        with pytest.raises(TypeError):
            GroupSynthesis("g", group_by="team", agent=agent)

    def test_fold_requires_prompt(self):
        agent = FakeAgent()
        with pytest.raises(TypeError):
            FoldSynthesis("f", label="out", agent=agent)


# ---------------------------------------------------------------------------
# Backward compatibility (agent=None)
# ---------------------------------------------------------------------------


class TestBackwardCompatNoAgent:
    def test_map_without_agent(self, mock_llm):
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", artifact_type="analysis")
        inp = _make_artifact("bio-alice", "Alice is an engineer.")
        results = t.execute([inp], {"llm_config": {}})

        assert len(results) == 1
        assert results[0].agent_fingerprint is None
        assert len(mock_llm) == 1

    def test_reduce_without_agent(self, mock_llm):
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out")
        inputs = [_make_artifact(f"a-{i}") for i in range(2)]
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert len(mock_llm) == 1

    def test_group_without_agent(self, mock_llm):
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis("s", group_by="team", prompt="{artifacts}")
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert len(mock_llm) == 1

    def test_fold_without_agent(self, mock_llm):
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis("fold", prompt="{accumulated}\n{artifact}", label="out")
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert len(mock_llm) == 1


# ---------------------------------------------------------------------------
# Fingerprint includes agent
# ---------------------------------------------------------------------------


class TestFingerprintWithAgent:
    def test_fingerprint_includes_agent_component(self):
        agent = FakeAgent(fingerprint="agent-fp-1")
        t = MapSynthesis("ws", prompt="x", agent=agent)
        fp = t.compute_fingerprint({})
        assert "agent" in fp.components
        assert fp.components["agent"] == "agent-fp-1"

    def test_fingerprint_changes_with_agent(self):
        agent_a = FakeAgent(fingerprint="fp-A")
        agent_b = FakeAgent(fingerprint="fp-B")
        t_a = MapSynthesis("ws", prompt="x", agent=agent_a)
        t_b = MapSynthesis("ws", prompt="x", agent=agent_b)
        fp_a = t_a.compute_fingerprint({})
        fp_b = t_b.compute_fingerprint({})
        assert fp_a.digest != fp_b.digest

    def test_fingerprint_without_agent_has_no_agent_component(self):
        t = MapSynthesis("ws", prompt="x")
        fp = t.compute_fingerprint({})
        assert "agent" not in fp.components

    def test_fingerprint_omits_model_when_agent_set(self):
        agent = FakeAgent()
        t = MapSynthesis("ws", prompt="x", agent=agent)
        fp = t.compute_fingerprint({"llm_config": {"model": "claude-sonnet-4-20250514"}})
        assert "agent" in fp.components
        assert "model" not in fp.components

    def test_reduce_fingerprint_includes_agent(self):
        agent = FakeAgent(fingerprint="reduce-fp")
        t = ReduceSynthesis("r", prompt="x", label="out", agent=agent)
        fp = t.compute_fingerprint({})
        assert "agent" in fp.components
        assert fp.components["agent"] == "reduce-fp"

    def test_group_fingerprint_includes_agent(self):
        agent = FakeAgent(fingerprint="group-fp")
        t = GroupSynthesis("g", group_by="team", prompt="x", agent=agent)
        fp = t.compute_fingerprint({})
        assert "agent" in fp.components
        assert fp.components["agent"] == "group-fp"

    def test_fold_fingerprint_includes_agent(self):
        agent = FakeAgent(fingerprint="fold-fp")
        t = FoldSynthesis("f", prompt="x", label="out", agent=agent)
        fp = t.compute_fingerprint({})
        assert "agent" in fp.components
        assert fp.components["agent"] == "fold-fp"


# ---------------------------------------------------------------------------
# Agent fingerprint on artifact
# ---------------------------------------------------------------------------


class TestAgentFingerprintOnArtifact:
    def test_map_artifact_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="artifact-fp")
        t = MapSynthesis("ws", prompt="{artifact}", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_fingerprint == "artifact-fp"

    def test_reduce_artifact_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="artifact-fp")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_fingerprint == "artifact-fp"

    def test_group_artifact_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="artifact-fp")
        t = GroupSynthesis("g", group_by="team", prompt="{artifacts}", agent=agent)
        results = t.execute([_make_artifact("a", team="alpha")], {"llm_config": {}})
        assert results[0].agent_fingerprint == "artifact-fp"

    def test_fold_artifact_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="artifact-fp")
        t = FoldSynthesis("f", prompt="{accumulated}\n{artifact}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_fingerprint == "artifact-fp"


# ---------------------------------------------------------------------------
# model_config=None when agent handles execution
# ---------------------------------------------------------------------------


class TestModelConfigNoneWithAgent:
    def test_map_model_config_none(self):
        agent = FakeAgent()
        t = MapSynthesis("ws", prompt="{artifact}", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {"model": "claude-sonnet-4-20250514"}})
        assert results[0].model_config is None

    def test_reduce_model_config_none(self):
        agent = FakeAgent()
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {"model": "claude-sonnet-4-20250514"}})
        assert results[0].model_config is None

    def test_group_model_config_none(self):
        agent = FakeAgent()
        t = GroupSynthesis("g", group_by="team", prompt="{artifacts}", agent=agent)
        results = t.execute([_make_artifact("a", team="x")], {"llm_config": {"model": "claude-sonnet-4-20250514"}})
        assert results[0].model_config is None

    def test_fold_model_config_none(self):
        agent = FakeAgent()
        t = FoldSynthesis("f", prompt="{accumulated}\n{artifact}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {"model": "claude-sonnet-4-20250514"}})
        assert results[0].model_config is None


# ---------------------------------------------------------------------------
# Custom agent implementation (protocol check)
# ---------------------------------------------------------------------------


class TestCustomAgentImplementation:
    def test_minimal_agent_protocol(self):
        """A minimal class satisfying the Agent protocol works with transforms."""

        class MinimalAgent:
            def write(self, request: AgentRequest) -> AgentResult:
                return AgentResult(content=f"processed: {request.prompt[:20]}")

            def fingerprint_value(self) -> str:
                return "minimal-v1"

        assert isinstance(MinimalAgent(), Agent)

        agent = MinimalAgent()
        t = MapSynthesis("ws", prompt="Do: {artifact}", agent=agent)
        results = t.execute([_make_artifact("a", "hello world")], {"llm_config": {}})

        assert results[0].content.startswith("processed:")
        assert results[0].agent_fingerprint == "minimal-v1"
