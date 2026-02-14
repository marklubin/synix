"""Unit tests for the CitationEnrichmentFixer."""

from __future__ import annotations

import json

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.fixers import (
    CitationEnrichmentFixer,
    FixContext,
    get_fixer,
)
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import Violation
from synix.core.models import Artifact, Pipeline

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


def _make_artifact(aid, atype="episode", content="test content", **meta):
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


class _MockSearchResult:
    def __init__(self, label, content):
        self.label = label
        self.content = content


class _MockSearchIndex:
    def __init__(self, results=None):
        self._results = results or []
        self.queries = []

    def query(self, q, **kwargs):
        self.queries.append(q)
        return self._results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_violation(label="core-1"):
    return Violation(
        violation_type="ungrounded_claim",
        severity="warning",
        message="Ungrounded claim: expert in quantum computing",
        label=label,
        field="content",
        metadata={
            "claim": "expert in quantum computing",
            "suggestion": "Link to conversation about computing background",
            "artifact_id": "hash1",
        },
        violation_id="vid-citation-1",
    )


# ---------------------------------------------------------------------------
# CitationEnrichmentFixer tests
# ---------------------------------------------------------------------------


class TestCitationEnrichmentFixer:
    def test_resolved_adds_citation(self, store, provenance):
        """Fixer resolves ungrounded claim by adding citation."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            "Mark is an expert in quantum computing.",
            layer_name="core",
            layer_level=3,
        )
        store.save_artifact(art, "core", 3)

        new_content = "Mark is an expert in quantum computing [ep-42](synix://ep-42)."
        response = json.dumps(
            {
                "status": "resolved",
                "action": "add_citation",
                "content": new_content,
                "explanation": "Added citation to source conversation",
            }
        )
        mock_llm = _MockLLMClient(response)
        mock_search = _MockSearchIndex([_MockSearchResult("ep-42", "User discussed quantum computing background")])

        ctx = FixContext(store, provenance, Pipeline("test"), search_index=mock_search, llm_client=mock_llm)

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation(), ctx)

        assert action.action == "rewrite"
        assert "synix://" in action.new_content
        assert action.new_artifact_id.startswith("sha256:")
        assert action.interactive is True
        assert action.evidence_source_ids == ["ep-42"]
        assert action.llm_explanation == "Added citation to source conversation"

    def test_unresolved_when_llm_cannot_find_source(self, store, provenance):
        """Fixer marks unresolved when LLM can't find a source."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            "Mark is an expert in quantum computing.",
            layer_name="core",
            layer_level=3,
        )
        store.save_artifact(art, "core", 3)

        response = json.dumps(
            {
                "status": "unresolved",
                "content": "",
                "explanation": "No source evidence found for this claim",
            }
        )
        mock_llm = _MockLLMClient(response)

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=mock_llm)

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation(), ctx)

        assert action.action == "unresolved"
        assert action.new_content == ""
        assert "No source evidence" in action.llm_explanation

    def test_missing_artifact_skip(self, store, provenance):
        """Fixer skips when artifact is not found in store."""
        ctx = FixContext(store, provenance, Pipeline("test"))

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation("nonexistent"), ctx)

        assert action.action == "skip"
        assert "not found" in action.description.lower()

    def test_no_llm_client_skip(self, store, provenance):
        """Fixer skips when no LLM client is available."""
        art = _make_artifact("core-1", "core_memory", "Some content.", layer_name="core", layer_level=3)
        store.save_artifact(art, "core", 3)

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=None)

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation(), ctx)

        assert action.action == "skip"
        assert "no llm client" in action.description.lower()

    def test_llm_error_skip(self, store, provenance):
        """Fixer skips when LLM raises an exception."""
        art = _make_artifact("core-1", "core_memory", "Some content.", layer_name="core", layer_level=3)
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient(RuntimeError("API error"))

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=mock_llm)

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation(), ctx)

        assert action.action == "skip"
        assert "LLM error" in action.description

    def test_invalid_json_response_unresolved(self, store, provenance):
        """Fixer returns unresolved when LLM response is not valid JSON."""
        art = _make_artifact("core-1", "core_memory", "Some content.", layer_name="core", layer_level=3)
        store.save_artifact(art, "core", 3)

        mock_llm = _MockLLMClient("not json at all")

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=mock_llm)

        fixer = CitationEnrichmentFixer()
        action = fixer.fix(_make_violation(), ctx)

        assert action.action == "unresolved"
        assert "parse" in action.description.lower()

    def test_registered_in_registry(self):
        fixer = get_fixer("citation_enrichment")
        assert isinstance(fixer, CitationEnrichmentFixer)
        assert fixer.interactive is True
        assert "ungrounded_claim" in fixer.handles_violation_types

    def test_can_handle_matches(self):
        fixer = CitationEnrichmentFixer()
        v_match = Violation("ungrounded_claim", "warning", "msg", "a", "content")
        v_no_match = Violation("semantic_conflict", "error", "msg", "a", "content")
        assert fixer.can_handle(v_match) is True
        assert fixer.can_handle(v_no_match) is False
