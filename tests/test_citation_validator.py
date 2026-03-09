"""Unit tests for the Citation validator."""

from __future__ import annotations

import json

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.build.validators import (
    Citation,
    ValidationContext,
)
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def build_dir(tmp_path):
    d = tmp_path / "build"
    d.mkdir()
    return d


def _make_artifact(aid, atype="episode", content="test", **meta):
    return Artifact(
        label=aid,
        artifact_type=atype,
        content=content,
        metadata=meta,
    )


def _build_store(build_dir, layer_artifacts, parent_labels_map=None):
    synix_dir = create_test_snapshot(build_dir, layer_artifacts, parent_labels_map=parent_labels_map)
    return SnapshotArtifactCache(synix_dir)


def _build_ctx(build_dir, layer_artifacts, parent_labels_map=None):
    store = _build_store(build_dir, layer_artifacts, parent_labels_map)
    return store, ValidationContext(store=store)


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
# Citation validator tests
# ---------------------------------------------------------------------------


class TestCitationValidator:
    def test_all_claims_cited_no_violations(self, build_dir):
        """LLM returns no ungrounded claims -> no violations."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark uses Python [ep-1](synix://ep-1).",
            layer_name="core",
        )
        store, ctx = _build_ctx(build_dir, {"core": [art]})

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        violations = validator.validate([art], ctx)
        assert violations == []
        assert len(mock_llm.calls) == 1

    def test_ungrounded_claims_produce_violations(self, build_dir):
        """LLM finds ungrounded claims -> violations with correct type and metadata."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark is an expert in quantum computing.",
            layer_name="core",
        )
        store, ctx = _build_ctx(build_dir, {"core": [art]})

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

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        violations = validator.validate([art], ctx)

        assert len(violations) == 1
        v = violations[0]
        assert v.violation_type == "ungrounded_claim"
        assert v.severity == "error"
        assert v.label == "core-1"
        assert v.field == "content"
        assert v.metadata["claim"] == "expert in quantum computing"
        assert v.metadata["suggestion"] == "Link to conversation about computing background"
        assert v.violation_id != ""

    def test_llm_error_produces_failure_violation(self, build_dir):
        """LLM raises an exception per-artifact -> no crash, produces citation_check_failed violation."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Some content here.",
            layer_name="core",
        )
        store, ctx = _build_ctx(build_dir, {"core": [art]})

        mock_llm = _MockLLMClient(RuntimeError("API unavailable"))

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "citation_check_failed"
        assert violations[0].severity == "error"
        assert violations[0].label == "core-1"

    def test_invalid_json_produces_failure_violation(self, build_dir):
        """Invalid JSON from LLM -> produces citation_check_failed violation."""
        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store, ctx = _build_ctx(build_dir, {"core": [art]})

        mock_llm = _MockLLMClient("this is not valid json")

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "citation_check_failed"

    def test_bad_llm_config_fails_closed(self, build_dir):
        """Invalid LLM config -> raises RuntimeError by default (fail closed)."""
        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store, ctx = _build_ctx(build_dir, {"core": [art]})

        validator = Citation(layers=[], llm_config={"provider": "nonexistent_provider_xyz"})
        with pytest.raises(RuntimeError, match="could not create LLM client"):
            validator.validate([art], ctx)

    def test_bad_llm_config_fail_open(self, build_dir):
        """With fail_open=True, bad LLM config returns empty."""
        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store, ctx = _build_ctx(build_dir, {"core": [art]})

        validator = Citation(layers=[], llm_config={"provider": "nonexistent_provider_xyz"}, fail_open=True)
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_missing_prompt_fails_closed(self, build_dir, tmp_path):
        """Missing prompt file raises RuntimeError by default."""
        import synix.build.validators as vmod

        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store, ctx = _build_ctx(build_dir, {"core": [art]})
        mock_llm = _MockLLMClient('{"ungrounded": []}')

        # Temporarily point prompts to a non-existent directory
        orig_file = vmod.__file__
        vmod.__file__ = str(tmp_path / "fake" / "validators.py")
        try:
            validator = Citation(layers=[])
            validator._llm_client = mock_llm
            with pytest.raises(RuntimeError, match="prompt not found"):
                validator.validate([art], ctx)
        finally:
            vmod.__file__ = orig_file

    def test_missing_prompt_fail_open(self, build_dir, tmp_path):
        """With fail_open=True, missing prompt returns empty."""
        import synix.build.validators as vmod

        art = _make_artifact("core-1", "core_memory", content="test", layer_name="core")
        store, ctx = _build_ctx(build_dir, {"core": [art]})
        mock_llm = _MockLLMClient('{"ungrounded": []}')

        orig_file = vmod.__file__
        vmod.__file__ = str(tmp_path / "fake" / "validators.py")
        try:
            validator = Citation(layers=[], fail_open=True)
            validator._llm_client = mock_llm
            violations = validator.validate([art], ctx)
            assert violations == []
        finally:
            vmod.__file__ = orig_file

    def test_max_artifacts_respected(self, build_dir):
        """Only first max_artifacts are checked when configured."""
        arts = [_make_artifact(f"core-{i}", "core_memory", content=f"content {i}", layer_name="core") for i in range(5)]
        store, ctx = _build_ctx(build_dir, {"core": arts})

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = Citation(layers=[], max_artifacts=2)
        validator._llm_client = mock_llm
        validator.validate(arts, ctx)
        assert len(mock_llm.calls) == 2

    def test_all_artifacts_checked_by_default(self, build_dir):
        """Without max_artifacts, all artifacts are checked."""
        arts = [_make_artifact(f"core-{i}", "core_memory", content=f"content {i}", layer_name="core") for i in range(5)]
        store, ctx = _build_ctx(build_dir, {"core": arts})

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        validator.validate(arts, ctx)
        assert len(mock_llm.calls) == 5

    def test_instantiation(self):
        v = Citation(layers=[])
        assert isinstance(v, Citation)

    def test_llm_trace_stored(self, build_dir):
        """LLM call produces a trace artifact in a writable store."""
        art = _make_artifact("core-1", "core_memory", content="test content", layer_name="core")
        # Use ArtifactStore for this test since it needs save_artifact for traces
        writable_store = ArtifactStore(build_dir)
        writable_store.save_artifact(art, "core", 3)
        ctx = ValidationContext(store=writable_store)

        mock_llm = _MockLLMClient('{"ungrounded": []}')

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        validator.validate([art], ctx)

        traces = writable_store.list_artifacts("traces")
        assert len(traces) >= 1
        assert traces[0].artifact_type == "llm_trace"

    def test_existing_citations_passed_to_prompt(self, build_dir):
        """Verify that existing synix:// citations in the artifact are noted."""
        art = _make_artifact(
            "core-1",
            "core_memory",
            content="Mark likes Python [ep-1](synix://ep-1). He also knows Rust.",
            layer_name="core",
        )
        store, ctx = _build_ctx(build_dir, {"core": [art]})

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

        validator = Citation(layers=[])
        validator._llm_client = mock_llm
        violations = validator.validate([art], ctx)

        assert len(violations) == 1
        assert "synix://ep-1" in violations[0].metadata["existing_citations"]
