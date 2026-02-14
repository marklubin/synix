"""Unit tests for the CitationValidator."""

from __future__ import annotations

import json

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import (
    CitationValidator,
    ValidationContext,
    get_validator,
)
from synix.core.models import Artifact

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def build_dir(tmp_path):
    d = tmp_path / "build"
    d.mkdir()
    return d


@pytest.fixture
def store(build_dir):
    return ArtifactStore(build_dir)


@pytest.fixture
def provenance(build_dir):
    return ProvenanceTracker(build_dir)


@pytest.fixture
def ctx(store, provenance):
    return ValidationContext(store=store, provenance=provenance)


def _make_artifact(aid, atype="episode", content="test", **meta):
    return Artifact(
        label=aid,
        artifact_type=atype,
        content=content,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, content):
        self.content = content


class _MockLLMClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def complete(self, messages, artifact_desc="", **kwargs):
        self.calls.append(messages)
        if isinstance(self.response_text, Exception):
            raise self.response_text
        return _MockResponse(self.response_text)


# ---------------------------------------------------------------------------
# CitationValidator tests
# ---------------------------------------------------------------------------


class TestCitationValidator:
    def test_all_claims_cited_no_violations(self, store, provenance, ctx):
        """LLM returns no ungrounded claims -> no violations."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark uses Python [ep-1](synix://ep-1).",
            layer_name="core",
        )
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        violations = validator.validate([art], ctx)
        assert violations == []
        assert len(mock_llm.calls) == 1

    def test_ungrounded_claims_produce_violations(self, store, provenance, ctx):
        """LLM finds ungrounded claims -> violations with correct type and metadata."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark is an expert in quantum computing.",
            layer_name="core",
        )
        store.save_artifact(art, "core", 3)

        response = json.dumps(
            {
                "ungrounded": [
                    {
                        "claim": "expert in quantum computing",
                        "suggestion": "Link to conversation about computing background",
                    }
                ]
            }
        )
        mock_llm = _MockLLMClient(response)

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        violations = validator.validate([art], ctx)

        assert len(violations) == 1
        v = violations[0]
        assert v.violation_type == "ungrounded_claim"
        assert v.severity == "warning"
        assert v.label == "core-1"
        assert v.field == "content"
        assert v.metadata["claim"] == "expert in quantum computing"
        assert v.metadata["suggestion"] == "Link to conversation about computing background"
        assert v.violation_id != ""

    def test_llm_error_graceful_skip(self, store, provenance, ctx):
        """LLM raises an exception -> no crash, no violations."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Some content here.",
            layer_name="core",
        )
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient(RuntimeError("API unavailable"))

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_invalid_json_returns_empty(self, store, provenance, ctx):
        """Invalid JSON from LLM -> no violations."""
        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient("this is not valid json")

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_no_llm_client_returns_empty(self, store, provenance, ctx):
        """No LLM config and no _llm_client -> returns empty (can't create client)."""
        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")

        validator = get_validator("citation")
        validator._config = {}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_max_artifacts_respected(self, store, provenance, ctx):
        """Only first max_artifacts are checked."""
        arts = []
        for i in range(5):
            a = _make_artifact(f"core-{i}", "core_memory", content=f"content {i}", layer_name="core")
            store.save_artifact(a, "core", 3)
            arts.append(a)

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm, "max_artifacts": 2}
        validator.validate(arts, ctx)
        assert len(mock_llm.calls) == 2

    def test_registered_in_registry(self):
        v = get_validator("citation")
        assert isinstance(v, CitationValidator)

    def test_llm_trace_stored(self, store, provenance, ctx):
        """LLM call produces a trace artifact."""
        art = _make_artifact("core-1", "core_memory", content="test content", layer_name="core")
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        validator.validate([art], ctx)

        traces = store.list_artifacts("traces")
        assert len(traces) >= 1
        assert traces[0].artifact_type == "llm_trace"

    def test_existing_citations_passed_to_prompt(self, store, provenance, ctx):
        """Verify that existing synix:// citations in the artifact are noted."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark likes Python [ep-1](synix://ep-1). He also knows Rust.",
            layer_name="core",
        )
        store.save_artifact(art, "core", 3)

        response = json.dumps(
            {
                "ungrounded": [
                    {
                        "claim": "He also knows Rust",
                        "suggestion": "Add citation for Rust knowledge",
                    }
                ]
            }
        )
        mock_llm = _MockLLMClient(response)

        validator = get_validator("citation")
        validator._config = {"_llm_client": mock_llm}
        violations = validator.validate([art], ctx)

        assert len(violations) == 1
        assert "synix://ep-1" in violations[0].metadata["existing_citations"]
