"""Unit tests for the fixer framework."""

from __future__ import annotations

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.fixers import (
    BaseFixer,
    CitationEnrichment,
    FixAction,
    FixContext,
    FixResult,
    SemanticEnrichment,
    _find_downstream_artifacts,
    apply_fix,
    run_fixers,
)
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import ValidationResult, Violation
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
# FixAction data model tests
# ---------------------------------------------------------------------------


class TestFixAction:
    def test_basic_fields(self):
        a = FixAction(
            label="art-1",
            action="rewrite",
            original_artifact_id="sha256:abc",
            new_content="new content",
            new_artifact_id="sha256:def",
            description="test fix",
        )
        assert a.label == "art-1"
        assert a.action == "rewrite"
        assert a.interactive is False
        assert a.llm_explanation == ""
        assert a.evidence_source_ids == []
        assert a.downstream_invalidated == []

    def test_interactive_flag(self):
        a = FixAction(
            label="art-1",
            action="rewrite",
            original_artifact_id="",
            new_content="",
            new_artifact_id="",
            description="",
            interactive=True,
        )
        assert a.interactive is True


class TestFixResult:
    def test_empty_result(self):
        r = FixResult()
        assert r.fixed_count == 0
        assert r.skipped_count == 0
        assert r.actions == []
        assert r.errors == []

    def test_fixed_count(self):
        r = FixResult(
            actions=[
                FixAction("a", "rewrite", "", "", "", ""),
                FixAction("b", "redact", "", "", "", ""),
                FixAction("c", "skip", "", "", "", ""),
            ]
        )
        assert r.fixed_count == 2
        assert r.skipped_count == 1

    def test_unresolved_counts_as_skipped(self):
        r = FixResult(
            actions=[
                FixAction("a", "unresolved", "", "", "", ""),
            ]
        )
        assert r.fixed_count == 0
        assert r.skipped_count == 1


# ---------------------------------------------------------------------------
# Fixer instantiation tests (replaces registry tests)
# ---------------------------------------------------------------------------


class TestFixerInstantiation:
    def test_semantic_enrichment(self):
        fixer = SemanticEnrichment()
        assert isinstance(fixer, SemanticEnrichment)
        assert fixer.interactive is True
        assert "semantic_conflict" in fixer.handles_violation_types

    def test_citation_enrichment(self):
        fixer = CitationEnrichment()
        assert isinstance(fixer, CitationEnrichment)
        assert fixer.interactive is True
        assert "ungrounded_claim" in fixer.handles_violation_types

    def test_can_handle_matches(self):
        fixer = SemanticEnrichment()
        v_match = Violation("semantic_conflict", "error", "msg", "art-1", "f")
        v_no_match = Violation("type_c", "error", "msg", "art-1", "f")
        assert fixer.can_handle(v_match) is True
        assert fixer.can_handle(v_no_match) is False


# ---------------------------------------------------------------------------
# run_fixers tests
# ---------------------------------------------------------------------------


class _InlineFixer(BaseFixer):
    """Test-only fixer for run_fixers tests."""

    def to_config_dict(self) -> dict:
        return {}


class _MatchFixer(_InlineFixer):
    name = "test_match_fixer"
    handles_violation_types = ["match_type"]

    def fix(self, violation, ctx):
        return FixAction(violation.label, "skip", "", "", "", "skipped")


class _ErrorFixer(_InlineFixer):
    name = "test_error_fixer"
    handles_violation_types = ["error_type"]

    def fix(self, violation, ctx):
        raise RuntimeError("fixer broke")


class _DownstreamFixer(_InlineFixer):
    name = "test_downstream_fixer"
    handles_violation_types = ["downstream_type"]

    def fix(self, violation, ctx):
        return FixAction(violation.label, "rewrite", "", "new", "", "fixed")


class TestRunFixers:
    def test_no_fixers(self, store, provenance):
        pipeline = Pipeline("test")
        vr = ValidationResult(violations=[Violation("test", "error", "msg", "a", "f")])
        result = run_fixers(vr, pipeline, store, provenance)
        assert result.actions == []
        assert result.fixers_run == []

    def test_fixer_matches_violations(self, store, provenance):
        pipeline = Pipeline("test")
        pipeline.add_fixer(_MatchFixer())

        vr = ValidationResult(
            violations=[
                Violation("match_type", "error", "msg", "art-1", "f"),
                Violation("other_type", "error", "msg", "art-2", "f"),
            ]
        )
        result = run_fixers(vr, pipeline, store, provenance)
        assert len(result.actions) == 1
        assert result.actions[0].label == "art-1"
        assert "test_match_fixer" in result.fixers_run

    def test_fixer_error_captured(self, store, provenance):
        pipeline = Pipeline("test")
        pipeline.add_fixer(_ErrorFixer())

        vr = ValidationResult(
            violations=[
                Violation("error_type", "error", "msg", "art-1", "f"),
            ]
        )
        result = run_fixers(vr, pipeline, store, provenance)
        assert len(result.errors) == 1
        assert "fixer broke" in result.errors[0]
        assert result.actions == []

    def test_downstream_invalidation_computed(self, store, provenance):
        """Verify that run_fixers populates downstream_invalidated on actions."""
        # Set up provenance: child depends on parent
        provenance.record("parent-1", parent_labels=[])
        provenance.record("child-1", parent_labels=["parent-1"])
        provenance.record("child-2", parent_labels=["parent-1"])

        pipeline = Pipeline("test")
        pipeline.add_fixer(_DownstreamFixer())

        vr = ValidationResult(
            violations=[
                Violation("downstream_type", "error", "msg", "parent-1", "f"),
            ]
        )
        result = run_fixers(vr, pipeline, store, provenance)
        assert len(result.actions) == 1
        assert set(result.actions[0].downstream_invalidated) == {"child-1", "child-2"}
        assert set(result.rebuild_required) == {"child-1", "child-2"}


# ---------------------------------------------------------------------------
# _find_downstream_artifacts tests
# ---------------------------------------------------------------------------


class TestFindDownstream:
    def test_finds_children(self, provenance):
        provenance.record("child-1", parent_labels=["parent-1"])
        provenance.record("child-2", parent_labels=["parent-1"])
        provenance.record("unrelated", parent_labels=["other"])

        downstream = _find_downstream_artifacts("parent-1", provenance)
        assert set(downstream) == {"child-1", "child-2"}

    def test_no_children(self, provenance):
        provenance.record("leaf", parent_labels=["some-parent"])
        downstream = _find_downstream_artifacts("leaf", provenance)
        assert downstream == []


# ---------------------------------------------------------------------------
# apply_fix tests
# ---------------------------------------------------------------------------


class TestApplyFix:
    def test_rewrites_artifact(self, store, provenance):
        art = _make_artifact("art-1", "episode", "original content", layer_name="episodes", layer_level=1)
        store.save_artifact(art, "episodes", 1)
        provenance.record("art-1", parent_labels=["t-1"])

        action = FixAction(
            label="art-1",
            action="rewrite",
            original_artifact_id=art.artifact_id,
            new_content="fixed content",
            new_artifact_id="sha256:new",
            description="test fix",
            evidence_source_ids=["evidence-1"],
        )
        apply_fix(action, store, provenance)

        reloaded = store.load_artifact("art-1")
        assert reloaded is not None
        assert reloaded.content == "fixed content"
        assert "sha256:" in reloaded.artifact_id

        # Provenance preserves original parents (evidence sources are
        # reference context, not true input lineage)
        parents = provenance.get_parents("art-1")
        assert "t-1" in parents

    def test_missing_artifact_noop(self, store, provenance):
        action = FixAction(
            label="nonexistent",
            action="rewrite",
            original_artifact_id="",
            new_content="new",
            new_artifact_id="",
            description="",
        )
        # Should not raise
        apply_fix(action, store, provenance)


# ---------------------------------------------------------------------------
# Mock helpers for SemanticEnrichment fixer
# ---------------------------------------------------------------------------


class _MockLLMResponse:
    def __init__(self, content):
        self.content = content


class _MockLLMClient:
    def __init__(self, response_content):
        self._response = response_content
        self.calls = []

    def complete(self, messages, artifact_desc="", **kwargs):
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response
        return _MockLLMResponse(self._response)


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
# SemanticEnrichment fixer tests
# ---------------------------------------------------------------------------


class TestSemanticEnrichmentFixer:
    def _make_violation(self, label="a-1"):
        return Violation(
            violation_type="semantic_conflict",
            severity="warning",
            message="Contradiction",
            label=label,
            field="content",
            metadata={
                "claim_a": "likes cats",
                "claim_b": "hates cats",
                "claim_a_source_hint": "conversation about pets",
                "claim_b_source_hint": "conversation about animals",
                "artifact_id": "hash1",
            },
            violation_id="vid-1",
        )

    def test_resolved_path(self, store, provenance):
        art = _make_artifact("a-1", "episode", "He likes cats. He hates cats.")
        store.save_artifact(art, "episodes", 1)

        mock_llm = _MockLLMClient(
            '{"status": "resolved", "content": "He initially liked cats but later changed his mind.", '
            '"explanation": "Temporal resolution"}'
        )
        mock_search = _MockSearchIndex(
            [
                _MockSearchResult("src-1", "User said they like cats"),
            ]
        )

        ctx = FixContext(store, provenance, Pipeline("test"), search_index=mock_search, llm_client=mock_llm)

        fixer = SemanticEnrichment()
        action = fixer.fix(self._make_violation(), ctx)

        assert action.action == "rewrite"
        assert "initially liked cats" in action.new_content
        assert action.evidence_source_ids == ["src-1"]
        assert action.interactive is True
        assert action.llm_explanation == "Temporal resolution"

    def test_unresolved_path(self, store, provenance):
        art = _make_artifact("a-1", "episode", "Contradictory content.")
        store.save_artifact(art, "episodes", 1)

        mock_llm = _MockLLMClient(
            '{"status": "unresolved", "content": "", "explanation": "Not enough context to resolve"}'
        )

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=mock_llm)

        fixer = SemanticEnrichment()
        action = fixer.fix(self._make_violation(), ctx)

        assert action.action == "unresolved"
        assert action.new_content == ""
        assert "Not enough context" in action.llm_explanation

    def test_missing_artifact_skip(self, store, provenance):
        ctx = FixContext(store, provenance, Pipeline("test"))
        fixer = SemanticEnrichment()
        action = fixer.fix(self._make_violation("nonexistent"), ctx)
        assert action.action == "skip"
        assert "not found" in action.description.lower()

    def test_no_llm_client_skip(self, store, provenance):
        art = _make_artifact("a-1", "episode", "Some content.")
        store.save_artifact(art, "episodes", 1)

        ctx = FixContext(store, provenance, Pipeline("test"), llm_client=None)

        fixer = SemanticEnrichment()
        action = fixer.fix(self._make_violation(), ctx)
        assert action.action == "skip"
        assert "no llm client" in action.description.lower()

    def test_evidence_provenance_on_apply(self, store, provenance):
        """Full flow: fix then apply, verify provenance includes evidence."""
        art = _make_artifact("a-1", "episode", "Original content.", layer_name="episodes", layer_level=1)
        store.save_artifact(art, "episodes", 1)
        provenance.record("a-1", parent_labels=["t-1"])

        mock_llm = _MockLLMClient('{"status": "resolved", "content": "Fixed content.", "explanation": "resolved"}')
        mock_search = _MockSearchIndex(
            [
                _MockSearchResult("evidence-1", "evidence text"),
            ]
        )

        ctx = FixContext(store, provenance, Pipeline("test"), search_index=mock_search, llm_client=mock_llm)

        fixer = SemanticEnrichment()
        action = fixer.fix(self._make_violation(), ctx)
        assert action.action == "rewrite"

        # Apply the fix
        apply_fix(action, store, provenance)

        reloaded = store.load_artifact("a-1")
        assert reloaded.content == "Fixed content."

        parents = provenance.get_parents("a-1")
        assert "t-1" in parents

    def test_semantic_enrichment_instantiation(self):
        fixer = SemanticEnrichment()
        assert isinstance(fixer, SemanticEnrichment)
        assert fixer.interactive is True
        assert "semantic_conflict" in fixer.handles_violation_types
