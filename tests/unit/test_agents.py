"""Tests for the Agent gateway interface."""

from __future__ import annotations

import pytest

from synix.agents import Agent, AgentRequest, AgentResult


class FakeAgent:
    """Minimal Agent protocol implementation for testing."""

    def __init__(self, response: str = "fake output", fingerprint: str = "fake-fp-001"):
        self._response = response
        self._fingerprint = fingerprint

    def write(self, request: AgentRequest) -> AgentResult:
        return AgentResult(content=self._response)

    def fingerprint_value(self) -> str:
        return self._fingerprint


class TestAgentRequest:
    def test_construction(self):
        req = AgentRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.max_tokens is None
        assert req.metadata == {}

    def test_with_metadata(self):
        req = AgentRequest(
            prompt="render this",
            max_tokens=500,
            metadata={"shape": "map", "input_labels": ["a", "b"]},
        )
        assert req.max_tokens == 500
        assert req.metadata["shape"] == "map"

    def test_frozen(self):
        req = AgentRequest(prompt="hello")
        with pytest.raises(AttributeError):
            req.prompt = "changed"


class TestAgentResult:
    def test_construction(self):
        result = AgentResult(content="output text")
        assert result.content == "output text"

    def test_frozen(self):
        result = AgentResult(content="output")
        with pytest.raises(AttributeError):
            result.content = "changed"


class TestAgentProtocol:
    def test_fake_agent_satisfies_protocol(self):
        agent = FakeAgent()
        assert isinstance(agent, Agent)

    def test_write_returns_result(self):
        agent = FakeAgent(response="test output")
        result = agent.write(AgentRequest(prompt="input"))
        assert result.content == "test output"

    def test_fingerprint_value(self):
        agent = FakeAgent(fingerprint="abc123")
        assert agent.fingerprint_value() == "abc123"

    def test_fingerprint_deterministic(self):
        agent = FakeAgent(fingerprint="stable")
        assert agent.fingerprint_value() == agent.fingerprint_value()

    def test_object_without_write_fails_isinstance(self):
        class NotAnAgent:
            def fingerprint_value(self) -> str:
                return "x"

        assert not isinstance(NotAnAgent(), Agent)

    def test_object_without_fingerprint_fails_isinstance(self):
        class NotAnAgent:
            def write(self, request):
                return AgentResult(content="x")

        assert not isinstance(NotAnAgent(), Agent)

    def test_different_fingerprints_for_different_agents(self):
        a1 = FakeAgent(fingerprint="fp-1")
        a2 = FakeAgent(fingerprint="fp-2")
        assert a1.fingerprint_value() != a2.fingerprint_value()


class TestArtifactAgentFingerprint:
    def test_artifact_has_agent_fingerprint_field(self):
        from synix.core.models import Artifact

        art = Artifact(label="test", artifact_type="test", content="hello")
        assert art.agent_fingerprint is None

    def test_artifact_with_agent_fingerprint(self):
        from synix.core.models import Artifact

        art = Artifact(
            label="test",
            artifact_type="test",
            content="hello",
            agent_fingerprint="agent-fp-xyz",
        )
        assert art.agent_fingerprint == "agent-fp-xyz"
