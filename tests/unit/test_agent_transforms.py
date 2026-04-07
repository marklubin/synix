"""Tests for Agent gateway integration in generic synthesis transforms."""

from __future__ import annotations

import pytest

from synix import Artifact
from synix.agents import Agent, Group
from synix.transforms import FoldSynthesis, GroupSynthesis, MapSynthesis, ReduceSynthesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Test double satisfying the Agent protocol."""

    def __init__(
        self, response: str = "agent output", fingerprint: str = "test-agent-fp", agent_id: str = "test-agent"
    ):
        self._response = response
        self._fingerprint = fingerprint
        self._agent_id = agent_id
        self.map_calls: list[tuple[Artifact, str]] = []
        self.reduce_calls: list[tuple[list[Artifact], str]] = []
        self.group_calls: list[tuple[list[Artifact], str]] = []
        self.fold_calls: list[tuple[str, Artifact, int, int, str]] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def fingerprint_value(self) -> str:
        return self._fingerprint

    def map(self, artifact: Artifact, task_prompt: str) -> str:
        self.map_calls.append((artifact, task_prompt))
        return self._response

    def reduce(self, artifacts: list[Artifact], task_prompt: str) -> str:
        self.reduce_calls.append((artifacts, task_prompt))
        return self._response

    def group(self, artifacts: list[Artifact], task_prompt: str) -> list[Group]:
        self.group_calls.append((artifacts, task_prompt))
        # Return one group per artifact for testing
        return [Group(key=a.label, artifacts=[a], content=self._response) for a in artifacts]

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int, task_prompt: str) -> str:
        self.fold_calls.append((accumulated, artifact, step, total, task_prompt))
        return self._response


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
    def test_agent_map_called_with_artifact_and_task_prompt(self):
        agent = FakeAgent()
        t = MapSynthesis(
            "ws",
            prompt="Analyze: {artifact}",
            artifact_type="analysis",
            agent=agent,
        )
        inp = _make_artifact("bio-alice", "Alice is an engineer.")
        t.execute([inp], {"llm_config": {}})

        assert len(agent.map_calls) == 1
        artifact, task_prompt = agent.map_calls[0]
        assert artifact.label == "bio-alice"
        assert artifact.content == "Alice is an engineer."
        assert "Alice is an engineer." in task_prompt
        assert task_prompt == "Analyze: Alice is an engineer."

    def test_artifact_has_agent_fingerprint(self):
        agent = FakeAgent(fingerprint="map-fp-123")
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].agent_fingerprint == "map-fp-123"

    def test_artifact_has_agent_id(self):
        agent = FakeAgent(agent_id="map-agent-1")
        t = MapSynthesis("ws", prompt="Analyze: {artifact}", agent=agent)
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].agent_id == "map-agent-1"

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
    def test_agent_reduce_called_with_task_prompt(self):
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

        assert len(agent.reduce_calls) == 1
        artifacts, task_prompt = agent.reduce_calls[0]
        # Reduce receives sorted artifacts
        assert len(artifacts) == 3
        assert results[0].agent_fingerprint == "test-agent-fp"
        assert results[0].agent_id == "test-agent"
        assert results[0].model_config is None
        # task_prompt contains rendered artifacts text
        assert "profile" in task_prompt
        assert task_prompt.startswith("Analyze: ")

    def test_artifact_content_from_agent(self):
        agent = FakeAgent(response="reduced output")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        inputs = [_make_artifact(f"a-{i}") for i in range(2)]
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].content == "reduced output"

    def test_artifact_has_agent_id(self):
        agent = FakeAgent(agent_id="reduce-agent-1")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        inputs = [_make_artifact(f"a-{i}") for i in range(2)]
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_id == "reduce-agent-1"


# ---------------------------------------------------------------------------
# GroupSynthesis + Agent
# ---------------------------------------------------------------------------


class TestGroupWithAgent:
    def test_agent_group_called_with_all_inputs_and_task_prompt(self):
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

        # Agent.group() called once with all inputs and task_prompt
        assert len(agent.group_calls) == 1
        artifacts, task_prompt = agent.group_calls[0]
        assert len(artifacts) == 3
        # task_prompt is rendered from the prompt template
        assert isinstance(task_prompt, str)
        # FakeAgent returns one group per artifact -> 3 results
        assert len(results) == 3

    def test_artifacts_have_agent_fingerprint_and_id(self):
        agent = FakeAgent(fingerprint="group-fp", agent_id="group-agent-1")
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "s",
            group_by="team",
            prompt="{artifacts}",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint == "group-fp"
        assert results[0].agent_id == "group-agent-1"
        assert results[0].model_config is None

    def test_group_label_uses_prefix(self):
        agent = FakeAgent()
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "s",
            group_by="team",
            prompt="{artifacts}",
            agent=agent,
            label_prefix="team",
        )
        results = t.execute(inputs, {"llm_config": {}})
        assert results[0].label == "team-ep-1"

    def test_group_label_without_prefix_uses_key(self):
        agent = FakeAgent()
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "s",
            group_by="team",
            prompt="{artifacts}",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})
        # FakeAgent returns group key = artifact label
        assert results[0].label == "ep-1"

    def test_group_metadata_contains_key_and_count(self):
        agent = FakeAgent()
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "s",
            group_by="team",
            prompt="{artifacts}",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})
        assert results[0].metadata["group_key"] == "ep-1"
        assert results[0].metadata["input_count"] == 1


# ---------------------------------------------------------------------------
# FoldSynthesis + Agent
# ---------------------------------------------------------------------------


class TestFoldWithAgent:
    def test_agent_called_per_step_with_task_prompt(self):
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
        assert len(agent.fold_calls) == 3
        assert len(results) == 1
        assert results[0].content == "accumulated"

        # Verify step/total and task_prompt in each call
        for i, (acc, art, step, total, task_prompt) in enumerate(agent.fold_calls):
            assert step == i + 1
            assert total == 3
            assert "Current:" in task_prompt
            assert "New:" in task_prompt

    def test_artifact_has_agent_fingerprint_and_id(self):
        agent = FakeAgent(fingerprint="fold-fp", agent_id="fold-agent-1")
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis(
            "fold",
            prompt="{accumulated}\n{artifact}",
            label="out",
            agent=agent,
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint == "fold-fp"
        assert results[0].agent_id == "fold-agent-1"
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

    def test_fold_initial_passed_to_first_step(self):
        agent = FakeAgent(response="step-result")
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis(
            "fold",
            prompt="{accumulated}\n{artifact}",
            initial="INITIAL VALUE",
            label="out",
            agent=agent,
        )
        t.execute(inputs, {"llm_config": {}})

        # First call should receive the initial value as accumulated
        acc, _art, step, total, task_prompt = agent.fold_calls[0]
        assert acc == "INITIAL VALUE"
        assert step == 1
        assert total == 1
        assert "INITIAL VALUE" in task_prompt


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


class TestEmptyFingerprintRejected:
    """RFC requires: empty fingerprint_value() must raise at construction."""

    def test_map_rejects_empty_fingerprint(self):
        class EmptyFpAgent:
            @property
            def agent_id(self):
                return "empty"

            def fingerprint_value(self):
                return ""

            def map(self, artifact, task_prompt):
                return "x"

            def reduce(self, artifacts, task_prompt):
                return "x"

            def group(self, artifacts, task_prompt):
                return []

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return "x"

        with pytest.raises(ValueError, match="empty fingerprint"):
            MapSynthesis("m", prompt="p", agent=EmptyFpAgent())

    def test_fold_rejects_empty_fingerprint(self):
        class EmptyFpAgent:
            @property
            def agent_id(self):
                return "empty"

            def fingerprint_value(self):
                return ""

            def map(self, artifact, task_prompt):
                return "x"

            def reduce(self, artifacts, task_prompt):
                return "x"

            def group(self, artifacts, task_prompt):
                return []

            def fold(self, accumulated, artifact, step, total, task_prompt):
                return "x"

        with pytest.raises(ValueError, match="empty fingerprint"):
            FoldSynthesis("f", prompt="p", label="out", agent=EmptyFpAgent())


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
        assert results[0].agent_id is None
        assert len(mock_llm) == 1

    def test_reduce_without_agent(self, mock_llm):
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out")
        inputs = [_make_artifact(f"a-{i}") for i in range(2)]
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert results[0].agent_id is None
        assert len(mock_llm) == 1

    def test_group_without_agent(self, mock_llm):
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis("s", group_by="team", prompt="{artifacts}")
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert results[0].agent_id is None
        assert len(mock_llm) == 1

    def test_fold_without_agent(self, mock_llm):
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis("fold", prompt="{accumulated}\n{artifact}", label="out")
        results = t.execute(inputs, {"llm_config": {}})

        assert results[0].agent_fingerprint is None
        assert results[0].agent_id is None
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
# Agent ID on artifact
# ---------------------------------------------------------------------------


class TestAgentIdOnArtifact:
    def test_map_artifact_agent_id(self):
        agent = FakeAgent(agent_id="map-agent")
        t = MapSynthesis("ws", prompt="{artifact}", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_id == "map-agent"

    def test_reduce_artifact_agent_id(self):
        agent = FakeAgent(agent_id="reduce-agent")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_id == "reduce-agent"

    def test_group_artifact_agent_id(self):
        agent = FakeAgent(agent_id="group-agent")
        t = GroupSynthesis("g", group_by="team", prompt="{artifacts}", agent=agent)
        results = t.execute([_make_artifact("a", team="alpha")], {"llm_config": {}})
        assert results[0].agent_id == "group-agent"

    def test_fold_artifact_agent_id(self):
        agent = FakeAgent(agent_id="fold-agent")
        t = FoldSynthesis("f", prompt="{accumulated}\n{artifact}", label="out", agent=agent)
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_id == "fold-agent"

    def test_no_agent_id_when_no_agent(self, mock_llm):
        t = MapSynthesis("ws", prompt="{artifact}")
        results = t.execute([_make_artifact("a")], {"llm_config": {}})
        assert results[0].agent_id is None


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
            @property
            def agent_id(self) -> str:
                return "minimal"

            def fingerprint_value(self) -> str:
                return "minimal-v1"

            def map(self, artifact: Artifact, task_prompt: str) -> str:
                return f"processed: {artifact.content[:20]}"

            def reduce(self, artifacts: list[Artifact], task_prompt: str) -> str:
                return "reduced"

            def group(self, artifacts: list[Artifact], task_prompt: str) -> list[Group]:
                return [Group(key="all", artifacts=artifacts, content="grouped")]

            def fold(self, accumulated: str, artifact: Artifact, step: int, total: int, task_prompt: str) -> str:
                return f"{accumulated}+{artifact.label}"

        assert isinstance(MinimalAgent(), Agent)

        agent = MinimalAgent()
        t = MapSynthesis("ws", prompt="Do: {artifact}", agent=agent)
        results = t.execute([_make_artifact("a", "hello world")], {"llm_config": {}})

        assert results[0].content.startswith("processed:")
        assert results[0].agent_fingerprint == "minimal-v1"
        assert results[0].agent_id == "minimal"
