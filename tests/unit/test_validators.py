"""Unit tests for the pluggable validator framework."""

from __future__ import annotations

import json

import pytest

from synix.build.snapshot_view import SnapshotArtifactCache
from synix.build.validators import (
    PII,
    BaseValidator,
    MutualExclusion,
    ProvenanceStep,
    RequiredField,
    SemanticConflict,
    ValidationContext,
    ValidationResult,
    Violation,
    ViolationQueue,
    _gather_artifacts,
    _parse_conflict_response,
    compute_violation_id,
    mutual_exclusion_violation,
    required_field_violation,
    run_validators,
)
from synix.core.models import Artifact, Pipeline
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
    """Create a SnapshotArtifactCache from layer->artifact mapping."""
    create_test_snapshot(build_dir, layer_artifacts, parent_labels_map=parent_labels_map)
    return SnapshotArtifactCache(build_dir / ".synix")


def _build_ctx(build_dir, layer_artifacts, parent_labels_map=None):
    """Create a ValidationContext backed by a snapshot."""
    store = _build_store(build_dir, layer_artifacts, parent_labels_map)
    return store, ValidationContext(store=store)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestViolation:
    def test_basic_fields(self):
        v = Violation(
            violation_type="mutual_exclusion",
            severity="error",
            message="test message",
            label="art-1",
            field="customer_id",
        )
        assert v.violation_type == "mutual_exclusion"
        assert v.severity == "error"
        assert v.label == "art-1"
        assert v.field == "customer_id"
        assert v.provenance_trace == []
        assert v.metadata == {}

    def test_with_provenance_trace(self):
        steps = [
            ProvenanceStep(label="a", layer="merge", field_value="acme"),
            ProvenanceStep(label="b", layer="episode", field_value="acme"),
        ]
        v = Violation(
            violation_type="test",
            severity="warning",
            message="msg",
            label="a",
            field="f",
            provenance_trace=steps,
        )
        assert len(v.provenance_trace) == 2
        assert v.provenance_trace[0].field_value == "acme"


class TestProvenanceStep:
    def test_defaults(self):
        s = ProvenanceStep(label="x", layer="episodes")
        assert s.field_value is None

    def test_with_value(self):
        s = ProvenanceStep(label="x", layer="episodes", field_value="acme")
        assert s.field_value == "acme"


class TestValidationResult:
    def test_empty_passes(self):
        r = ValidationResult()
        assert r.passed is True
        assert r.violations == []
        assert r.validators_run == []

    def test_with_error_fails(self):
        r = ValidationResult(
            violations=[
                Violation(
                    violation_type="test",
                    severity="error",
                    message="msg",
                    label="a",
                    field="f",
                )
            ],
            validators_run=["test"],
        )
        assert r.passed is False

    def test_warning_only_passes(self):
        r = ValidationResult(
            violations=[
                Violation(
                    violation_type="test",
                    severity="warning",
                    message="msg",
                    label="a",
                    field="f",
                )
            ],
            validators_run=["test"],
        )
        assert r.passed is True

    def test_to_dict(self):
        r = ValidationResult(
            violations=[
                Violation(
                    violation_type="mutual_exclusion",
                    severity="error",
                    message="test",
                    label="a",
                    field="customer_id",
                    metadata={"conflicting_values": ["x", "y"]},
                    provenance_trace=[
                        ProvenanceStep("a", "merge", "['x', 'y']"),
                    ],
                )
            ],
            validators_run=["mutual_exclusion"],
        )
        d = r.to_dict()
        assert d["passed"] is False
        assert len(d["violations"]) == 1
        assert d["violations"][0]["violation_type"] == "mutual_exclusion"
        assert d["violations"][0]["provenance_trace"][0]["label"] == "a"
        assert "mutual_exclusion" in d["validators_run"]


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_mutual_exclusion_violation(self):
        v = mutual_exclusion_violation("art-1", "customer_id", ["acme", "globex"])
        assert v.violation_type == "mutual_exclusion"
        assert v.severity == "error"
        assert v.label == "art-1"
        assert v.field == "customer_id"
        assert v.metadata["conflicting_values"] == ["acme", "globex"]
        assert "customer_id" in v.message
        assert "acme" in v.message

    def test_mutual_exclusion_custom_message(self):
        v = mutual_exclusion_violation("a", "f", ["x"], message="custom msg")
        assert v.message == "custom msg"

    def test_mutual_exclusion_warning_severity(self):
        v = mutual_exclusion_violation("a", "f", ["x"], severity="warning")
        assert v.severity == "warning"

    def test_required_field_violation(self):
        v = required_field_violation("art-2", "customer_id")
        assert v.violation_type == "required_field"
        assert v.severity == "error"
        assert v.label == "art-2"
        assert v.field == "customer_id"
        assert "customer_id" in v.message

    def test_required_field_custom_message(self):
        v = required_field_violation("a", "f", message="custom")
        assert v.message == "custom"


# ---------------------------------------------------------------------------
# Instantiation tests (replaces registry tests)
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_mutual_exclusion(self):
        v = MutualExclusion(field="customer_id", scope="merge", layers=[])
        assert isinstance(v, MutualExclusion)

    def test_required_field(self):
        v = RequiredField(field="customer_id", layers=[])
        assert isinstance(v, RequiredField)

    def test_pii(self):
        v = PII(layers=[])
        assert isinstance(v, PII)

    def test_semantic_conflict(self):
        v = SemanticConflict()
        assert isinstance(v, SemanticConflict)


# ---------------------------------------------------------------------------
# ValidationContext.trace_field_origin tests
# ---------------------------------------------------------------------------


class TestTraceFieldOrigin:
    def test_single_artifact_no_parents(self, build_dir):
        art = _make_artifact("t-1", "transcript", customer_id="acme", layer_name="transcripts")
        store, ctx = _build_ctx(build_dir, {"transcripts": [art]})

        steps = ctx.trace_field_origin("t-1", "customer_id")
        assert len(steps) == 1
        assert steps[0].label == "t-1"
        assert steps[0].field_value == "acme"
        assert steps[0].layer == "transcripts"

    def test_two_level_chain(self, build_dir):
        t = _make_artifact("t-1", "transcript", customer_id="acme", layer_name="transcripts")
        ep = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        store, ctx = _build_ctx(
            build_dir,
            {"transcripts": [t], "episodes": [ep]},
            parent_labels_map={"ep-1": ["t-1"]},
        )

        steps = ctx.trace_field_origin("ep-1", "customer_id")
        assert len(steps) == 2
        ids = [s.label for s in steps]
        assert "ep-1" in ids
        assert "t-1" in ids

    def test_missing_field_returns_none_value(self, build_dir):
        art = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store, ctx = _build_ctx(build_dir, {"transcripts": [art]})

        steps = ctx.trace_field_origin("t-1", "customer_id")
        assert len(steps) == 1
        assert steps[0].field_value is None

    def test_branching_provenance(self, build_dir):
        """Merge artifact with two parents from different customers."""
        t1 = _make_artifact("t-1", "transcript", customer_id="acme", layer_name="transcripts")
        t2 = _make_artifact("t-2", "transcript", customer_id="globex", layer_name="transcripts")
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex", layer_name="episodes")
        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")

        store, ctx = _build_ctx(
            build_dir,
            {"transcripts": [t1, t2], "episodes": [ep1, ep2], "merge": [merge]},
            parent_labels_map={
                "ep-1": ["t-1"],
                "ep-2": ["t-2"],
                "merge-ep-1": ["ep-1", "ep-2"],
            },
        )

        steps = ctx.trace_field_origin("merge-ep-1", "customer_id")
        ids = {s.label for s in steps}
        assert "merge-ep-1" in ids
        assert "ep-1" in ids
        assert "ep-2" in ids
        assert "t-1" in ids
        assert "t-2" in ids


# ---------------------------------------------------------------------------
# MutualExclusion validator tests
# ---------------------------------------------------------------------------


class TestMutualExclusionValidator:
    def test_no_violations_single_customer(self, build_dir):
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="acme", layer_name="episodes")
        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")

        store, ctx = _build_ctx(
            build_dir,
            {"episodes": [ep1, ep2], "merge": [merge]},
            parent_labels_map={"merge-ep-1": ["ep-1", "ep-2"]},
        )

        validator = MutualExclusion(field="customer_id", scope="merge", layers=[])
        violations = validator.validate([merge], ctx)
        assert violations == []

    def test_violation_multiple_customers(self, build_dir):
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex", layer_name="episodes")
        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")

        store, ctx = _build_ctx(
            build_dir,
            {"episodes": [ep1, ep2], "merge": [merge]},
            parent_labels_map={"merge-ep-1": ["ep-1", "ep-2"]},
        )

        validator = MutualExclusion(field="customer_id", scope="merge", layers=[])
        violations = validator.validate([merge], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "mutual_exclusion"
        assert violations[0].label == "merge-ep-1"
        assert "acme" in violations[0].metadata["conflicting_values"]
        assert "globex" in violations[0].metadata["conflicting_values"]

    def test_source_field_metadata_detected(self, build_dir):
        """source_customer_ids in merge artifact metadata also triggers."""
        merge = _make_artifact(
            "merge-ep-2",
            "merge",
            layer_name="merge",
            source_customer_ids=["acme", "globex"],
        )
        store, ctx = _build_ctx(build_dir, {"merge": [merge]})

        validator = MutualExclusion(field="customer_id", scope="merge", layers=[])
        violations = validator.validate([merge], ctx)
        assert len(violations) == 1

    def test_no_parents_no_violation(self, build_dir):
        """Artifact with no parents and no source metadata: no violation."""
        merge = _make_artifact("merge-ep-3", "merge", layer_name="merge")
        store, ctx = _build_ctx(build_dir, {"merge": [merge]})

        validator = MutualExclusion(field="customer_id", scope="merge", layers=[])
        violations = validator.validate([merge], ctx)
        assert violations == []

    def test_multiple_merge_artifacts(self, build_dir):
        """Only violating merge artifacts are flagged."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex", layer_name="episodes")
        ep3 = _make_artifact("ep-3", "episode", customer_id="acme", layer_name="episodes")
        m1 = _make_artifact("merge-1", "merge", layer_name="merge")
        m2 = _make_artifact("merge-2", "merge", layer_name="merge")

        store, ctx = _build_ctx(
            build_dir,
            {"episodes": [ep1, ep2, ep3], "merge": [m1, m2]},
            parent_labels_map={
                "merge-1": ["ep-1", "ep-2"],
                "merge-2": ["ep-1", "ep-3"],
            },
        )

        validator = MutualExclusion(field="customer_id", scope="merge", layers=[])
        violations = validator.validate([m1, m2], ctx)
        assert len(violations) == 1
        assert violations[0].label == "merge-1"


# ---------------------------------------------------------------------------
# RequiredField validator tests
# ---------------------------------------------------------------------------


class TestRequiredFieldValidator:
    def test_no_violations_field_present(self, build_dir):
        art = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        store, ctx = _build_ctx(build_dir, {"episodes": [art]})

        validator = RequiredField(field="customer_id", layers=[])
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_violation_field_missing(self, build_dir):
        art = _make_artifact("ep-1", "episode", layer_name="episodes")
        store, ctx = _build_ctx(build_dir, {"episodes": [art]})

        validator = RequiredField(field="customer_id", layers=[])
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "required_field"
        assert violations[0].label == "ep-1"

    def test_violation_field_empty_string(self, build_dir):
        art = _make_artifact("ep-1", "episode", customer_id="", layer_name="episodes")
        store, ctx = _build_ctx(build_dir, {"episodes": [art]})

        validator = RequiredField(field="customer_id", layers=[])
        violations = validator.validate([art], ctx)
        assert len(violations) == 1

    def test_violation_field_whitespace(self, build_dir):
        art = _make_artifact("ep-1", "episode", customer_id="  ", layer_name="episodes")
        store, ctx = _build_ctx(build_dir, {"episodes": [art]})

        validator = RequiredField(field="customer_id", layers=[])
        violations = validator.validate([art], ctx)
        assert len(violations) == 1

    def test_multiple_artifacts_mixed(self, build_dir):
        a1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        a2 = _make_artifact("ep-2", "episode", layer_name="episodes")
        a3 = _make_artifact("ep-3", "episode", customer_id="globex", layer_name="episodes")
        store, ctx = _build_ctx(build_dir, {"episodes": [a1, a2, a3]})

        validator = RequiredField(field="customer_id", layers=[])
        violations = validator.validate([a1, a2, a3], ctx)
        assert len(violations) == 1
        assert violations[0].label == "ep-2"


# ---------------------------------------------------------------------------
# _gather_artifacts tests
# ---------------------------------------------------------------------------


class TestGatherArtifacts:
    def test_filter_by_layers(self, build_dir):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store = _build_store(build_dir, {"episodes": [a1], "transcripts": [a2]})

        result = _gather_artifacts(store, {"layers": ["episodes"]})
        assert len(result) == 1
        assert result[0].label == "ep-1"

    def test_filter_by_scope_prefix(self, build_dir):
        a1 = _make_artifact("merge-1", "merge", layer_name="merge")
        a2 = _make_artifact("ep-1", "episode", layer_name="episodes")
        store = _build_store(build_dir, {"merge": [a1], "episodes": [a2]})

        result = _gather_artifacts(store, {"scope": "merge"})
        assert len(result) == 1
        assert result[0].label == "merge-1"

    def test_no_filter_returns_all(self, build_dir):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store = _build_store(build_dir, {"episodes": [a1], "transcripts": [a2]})

        result = _gather_artifacts(store, {})
        assert len(result) == 2

    def test_filter_multiple_layers(self, build_dir):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        a3 = _make_artifact("m-1", "merge", layer_name="merge")
        store = _build_store(build_dir, {"episodes": [a1], "transcripts": [a2], "merge": [a3]})

        result = _gather_artifacts(store, {"layers": ["episodes", "merge"]})
        labels = {a.label for a in result}
        assert labels == {"ep-1", "m-1"}


# ---------------------------------------------------------------------------
# run_validators integration tests
# ---------------------------------------------------------------------------


class TestRunValidators:
    def test_no_validators_empty_result(self, build_dir):
        store = _build_store(build_dir, {})
        pipeline = Pipeline("test")
        result = run_validators(pipeline, store)
        assert result.passed is True
        assert result.validators_run == []
        assert result.violations == []

    def test_mutual_exclusion_end_to_end(self, build_dir):
        """Full integration: pipeline declares mutual_exclusion validator."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex", layer_name="episodes")
        merge = _make_artifact("merge-1", "merge", layer_name="merge")

        store = _build_store(
            build_dir,
            {"episodes": [ep1, ep2], "merge": [merge]},
            parent_labels_map={"merge-1": ["ep-1", "ep-2"]},
        )

        pipeline = Pipeline("test")
        pipeline.add_validator(MutualExclusion(field="customer_id", scope="merge", layers=[]))

        result = run_validators(pipeline, store)
        assert result.passed is False
        assert "mutual_exclusion" in result.validators_run
        assert len(result.violations) == 1
        assert result.violations[0].label == "merge-1"

    def test_required_field_end_to_end(self, build_dir):
        """Full integration: pipeline declares required_field validator."""
        a1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        a2 = _make_artifact("ep-2", "episode", layer_name="episodes")

        store = _build_store(build_dir, {"episodes": [a1, a2]})

        pipeline = Pipeline("test")
        pipeline.add_validator(RequiredField(field="customer_id", layers=[]))

        result = run_validators(pipeline, store)
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].label == "ep-2"

    def test_multiple_validators(self, build_dir):
        """Both validators run; results aggregated."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", layer_name="episodes")

        store = _build_store(build_dir, {"episodes": [ep1, ep2]})

        pipeline = Pipeline("test")
        pipeline.add_validator(RequiredField(field="customer_id", layers=[]))
        pipeline.add_validator(MutualExclusion(field="customer_id", scope="merge", layers=[]))

        result = run_validators(pipeline, store)
        assert len(result.validators_run) == 2
        # Only required_field should have a violation (no merge artifacts)
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "required_field"

    def test_provenance_auto_resolved(self, build_dir):
        """Violations get provenance traces auto-resolved."""
        t1 = _make_artifact("t-1", "transcript", customer_id="acme", layer_name="transcripts")
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex", layer_name="episodes")
        merge = _make_artifact("merge-1", "merge", layer_name="merge")

        store = _build_store(
            build_dir,
            {"transcripts": [t1], "episodes": [ep1, ep2], "merge": [merge]},
            parent_labels_map={
                "ep-1": ["t-1"],
                "merge-1": ["ep-1", "ep-2"],
            },
        )

        pipeline = Pipeline("test")
        pipeline.add_validator(MutualExclusion(field="customer_id", scope="merge", layers=[]))

        result = run_validators(pipeline, store)
        assert len(result.violations) == 1
        trace = result.violations[0].provenance_trace
        assert len(trace) >= 2  # at least merge-1 and parents
        trace_ids = {s.label for s in trace}
        assert "merge-1" in trace_ids
        assert "ep-1" in trace_ids

    def test_all_pass(self, build_dir):
        """No violations when all artifacts are valid."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme", layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="acme", layer_name="episodes")

        store = _build_store(build_dir, {"episodes": [ep1, ep2]})

        pipeline = Pipeline("test")
        pipeline.add_validator(RequiredField(field="customer_id", layers=[]))

        result = run_validators(pipeline, store)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_run_validators_sorts_artifacts_and_violations_stably(self, build_dir):
        left = _make_artifact("z-art", content="left", layer_name="left")
        right = _make_artifact("a-art", content="right", layer_name="right")

        store = _build_store(build_dir, {"left": [left], "right": [right]})

        class _UnorderedValidator(BaseValidator):
            name = "unordered"

            def to_config_dict(self) -> dict:
                return {"layers": ["left", "right"]}

            def validate(self, artifacts, ctx):
                assert [artifact.label for artifact in artifacts] == ["z-art", "a-art"]
                return [
                    Violation(
                        "ungrounded_claim",
                        "error",
                        "msg z",
                        "z-art",
                        "content",
                        metadata={"claim": "zeta"},
                    ),
                    Violation(
                        "ungrounded_claim",
                        "error",
                        "msg a",
                        "a-art",
                        "content",
                        metadata={"claim": "alpha"},
                    ),
                ]

        pipeline = Pipeline("test")
        pipeline.add_validator(_UnorderedValidator())

        result = run_validators(pipeline, store)

        assert [(v.label, v.metadata["claim"]) for v in result.violations] == [
            ("z-art", "zeta"),
            ("a-art", "alpha"),
        ]
        assert [(v["label"], v["metadata"]["claim"]) for v in result.to_dict()["violations"]] == [
            ("a-art", "alpha"),
            ("z-art", "zeta"),
        ]


# ---------------------------------------------------------------------------
# Validator instance tests (replaces ValidatorDecl model tests)
# ---------------------------------------------------------------------------


class TestValidatorInstances:
    def test_mutual_exclusion_basic(self):
        v = MutualExclusion(field="customer_id", scope="merge", layers=[])
        assert v.name == "mutual_exclusion"

    def test_mutual_exclusion_config_dict(self):
        v = MutualExclusion(field="customer_id", scope="merge", layers=[])
        config = v.to_config_dict()
        assert config["field"] == "customer_id"

    def test_pipeline_add_validator(self):
        p = Pipeline("test")
        assert p.validators == []
        p.add_validator(MutualExclusion(field="customer_id", scope="merge", layers=[]))
        assert len(p.validators) == 1

    def test_pipeline_multiple_validators(self):
        p = Pipeline("test")
        p.add_validator(MutualExclusion(field="customer_id", scope="merge", layers=[]))
        p.add_validator(RequiredField(field="customer_id", layers=[]))
        assert len(p.validators) == 2


# ---------------------------------------------------------------------------
# RunResult.validation integration test
# ---------------------------------------------------------------------------


class TestRunResultValidation:
    def test_run_result_has_validation_field(self):
        from synix.build.runner import RunResult

        r = RunResult()
        assert r.validation is None


# ---------------------------------------------------------------------------
# compute_violation_id tests
# ---------------------------------------------------------------------------


class TestComputeViolationId:
    def test_deterministic(self):
        id1 = compute_violation_id("pii", "art-1", "claim a", "claim b")
        id2 = compute_violation_id("pii", "art-1", "claim a", "claim b")
        assert id1 == id2

    def test_normalizes_claims(self):
        id1 = compute_violation_id("pii", "art-1", "  Claim A  ", "Claim B")
        id2 = compute_violation_id("pii", "art-1", "claim a", "claim b")
        assert id1 == id2

    def test_sorted_claims(self):
        """Order of claims should not matter."""
        id1 = compute_violation_id("pii", "art-1", "claim b", "claim a")
        id2 = compute_violation_id("pii", "art-1", "claim a", "claim b")
        assert id1 == id2

    def test_different_type_different_id(self):
        id1 = compute_violation_id("pii", "art-1", "a", "b")
        id2 = compute_violation_id("semantic", "art-1", "a", "b")
        assert id1 != id2

    def test_different_artifact_different_id(self):
        id1 = compute_violation_id("pii", "art-1", "a", "b")
        id2 = compute_violation_id("pii", "art-2", "a", "b")
        assert id1 != id2

    def test_length_is_16(self):
        vid = compute_violation_id("pii", "art-1")
        assert len(vid) == 16


# ---------------------------------------------------------------------------
# PII validator tests
# ---------------------------------------------------------------------------


class TestPIIValidator:
    @pytest.fixture
    def ctx(self, build_dir):
        store = _build_store(build_dir, {})
        return ValidationContext(store=store)

    def test_detects_credit_card(self, ctx):
        art = _make_artifact("a-1", content="My card is 4111-1111-1111-1111 ok?")
        violations = PII(layers=[]).validate([art], ctx)
        cc_violations = [v for v in violations if v.metadata["pattern"] == "credit_card"]
        assert len(cc_violations) == 1
        assert cc_violations[0].violation_type == "pii"
        assert "4111" in cc_violations[0].metadata["redacted_value"]
        assert "****" in cc_violations[0].metadata["redacted_value"]

    def test_detects_ssn(self, ctx):
        art = _make_artifact("a-1", content="SSN: 123-45-6789")
        violations = PII(layers=[]).validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "ssn"
        assert "6789" in violations[0].metadata["redacted_value"]
        assert violations[0].metadata["redacted_value"].startswith("***-**-")

    def test_detects_email(self, ctx):
        art = _make_artifact("a-1", content="Email me at user@example.com")
        violations = PII(layers=[]).validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "email"
        assert violations[0].metadata["redacted_value"] == "us***@example.com"

    def test_detects_phone(self, ctx):
        art = _make_artifact("a-1", content="Call 555-867-5309")
        violations = PII(layers=[]).validate([art], ctx)
        assert len(violations) >= 1
        phone_violations = [v for v in violations if v.metadata["pattern"] == "phone"]
        assert len(phone_violations) >= 1
        assert "5309" in phone_violations[0].metadata["redacted_value"]

    def test_clean_text_no_violations(self, ctx):
        art = _make_artifact("a-1", content="This is clean text with no PII")
        violations = PII(layers=[]).validate([art], ctx)
        assert violations == []

    def test_multiple_pii_in_one_artifact(self, ctx):
        art = _make_artifact(
            "a-1",
            content="Card: 4111-1111-1111-1111 and SSN: 123-45-6789",
        )
        violations = PII(layers=[]).validate([art], ctx)
        patterns = {v.metadata["pattern"] for v in violations}
        assert "credit_card" in patterns
        assert "ssn" in patterns

    def test_patterns_config_filter(self, ctx):
        art = _make_artifact(
            "a-1",
            content="Card: 4111-1111-1111-1111 and SSN: 123-45-6789",
        )
        validator = PII(layers=[], patterns=["ssn"])
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "ssn"

    def test_violation_has_violation_id(self, ctx):
        art = _make_artifact("a-1", content="SSN: 123-45-6789")
        violations = PII(layers=[]).validate([art], ctx)
        assert violations[0].violation_id != ""

    def test_pii_instantiation(self):
        v = PII(layers=[])
        assert isinstance(v, PII)


# ---------------------------------------------------------------------------
# ViolationQueue tests
# ---------------------------------------------------------------------------


class TestViolationQueue:
    def test_save_load_roundtrip(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.save_state()

        q2 = ViolationQueue.load(build_dir)
        assert len(q2.active()) == 1
        assert q2.active()[0]["violation_id"] == "vid-1"

    def test_upsert_new(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        assert len(q.active()) == 1

    def test_upsert_existing_stays_active(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v1 = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v1)
        # Upsert again with same hash
        v2 = Violation(
            violation_type="pii",
            severity="warning",
            message="test updated",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v2)
        assert len(q.active()) == 1

    def test_ignore(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert len(q.active()) == 0

    def test_is_ignored(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert q.is_ignored("vid-1", "hash1") is True

    def test_is_ignored_invalidated_on_hash_change(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert q.is_ignored("vid-1", "hash1") is True
        assert q.is_ignored("vid-1", "hash2") is False

    def test_is_ignored_unknown_id(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        assert q.is_ignored("nonexistent", "hash1") is False

    def test_active_filters(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v1 = Violation(
            violation_type="pii",
            severity="warning",
            message="test1",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        v2 = Violation(
            violation_type="pii",
            severity="warning",
            message="test2",
            label="a-2",
            field="content",
            violation_id="vid-2",
            metadata={"artifact_id": "hash2"},
        )
        q.upsert(v1)
        q.upsert(v2)
        q.ignore("vid-1")
        active = q.active()
        assert len(active) == 1
        assert active[0]["violation_id"] == "vid-2"

    def test_resolve(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.resolve("vid-1", fix_action="redacted")
        assert len(q.active()) == 0

    def test_append_log_writes_jsonl(self, build_dir):
        import json

        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        log_path = build_dir / "violations.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        event = json.loads(lines[0])
        assert event["event"] == "detected"
        assert event["violation_id"] == "vid-1"

    def test_upsert_reactivates_ignored_on_hash_change(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert len(q.active()) == 0

        # Upsert with different hash should reactivate
        v2 = Violation(
            violation_type="pii",
            severity="warning",
            message="test",
            label="a-1",
            field="content",
            violation_id="vid-1",
            metadata={"artifact_id": "hash2"},
        )
        q.upsert(v2)
        assert len(q.active()) == 1


# ---------------------------------------------------------------------------
# _parse_conflict_response tests
# ---------------------------------------------------------------------------


class TestParseConflictResponse:
    def test_valid_json(self):
        resp = '{"conflicts": [{"claim_a": "A", "claim_b": "B"}]}'
        result = _parse_conflict_response(resp)
        assert len(result) == 1
        assert result[0]["claim_a"] == "A"

    def test_json_in_code_block(self):
        resp = '```json\n{"conflicts": [{"claim_a": "X", "claim_b": "Y"}]}\n```'
        result = _parse_conflict_response(resp)
        assert len(result) == 1
        assert result[0]["claim_a"] == "X"

    def test_empty_conflicts(self):
        resp = '{"conflicts": []}'
        result = _parse_conflict_response(resp)
        assert result == []

    def test_invalid_json_raises(self):
        resp = "this is not json at all"
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_conflict_response(resp)

    def test_missing_source_hints_defaulted(self):
        resp = '{"conflicts": [{"claim_a": "A", "claim_b": "B"}]}'
        result = _parse_conflict_response(resp)
        assert result[0]["claim_a_source_hint"] == ""
        assert result[0]["claim_b_source_hint"] == ""

    def test_conflicts_not_a_list_raises(self):
        resp = '{"conflicts": "not a list"}'
        with pytest.raises(ValueError, match="not a list"):
            _parse_conflict_response(resp)


# ---------------------------------------------------------------------------
# SemanticConflict validator tests
# ---------------------------------------------------------------------------


def _mock_llm_for_validator(monkeypatch, response_content):
    """Set up monkeypatches so SemanticConflict can create an LLM client.

    Returns a mock client instance for inspection (e.g. call_count).
    """
    call_count = [0]

    class _MockResponse:
        def __init__(self, content):
            self.content = content

    class _MockClient:
        def complete(self, messages=None, artifact_desc="", **kwargs):
            call_count[0] += 1
            if isinstance(response_content, Exception):
                raise response_content
            return _MockResponse(response_content)

    mock_client = _MockClient()
    # Patch at the source modules since they're imported locally in validate()
    monkeypatch.setattr("synix.build.llm_client.LLMClient", lambda cfg: mock_client)
    monkeypatch.setattr("synix.core.config.LLMConfig.from_dict", lambda d: None)
    return mock_client, call_count


class TestSemanticConflictValidator:
    @pytest.fixture
    def ctx(self, build_dir):
        store = _build_store(build_dir, {})
        return ValidationContext(store=store)

    def test_no_conflicts(self, ctx, monkeypatch):
        """LLM returns no conflicts -> no violations."""
        art = _make_artifact("monthly-1", "monthly", content="Mark likes Python.", layer_name="monthly")

        _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        validator = SemanticConflict(llm_config={"api_key": "test"})
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_conflict_found(self, ctx, monkeypatch):
        """LLM finds a conflict -> violation with metadata."""
        art = _make_artifact(
            "monthly-1", "monthly", content="Mark owns a BMW. Mark drives a Dodge Neon daily.", layer_name="monthly"
        )

        conflict_json = json.dumps(
            {
                "conflicts": [
                    {
                        "claim_a": "owns a BMW",
                        "claim_b": "drives a Dodge Neon daily",
                        "claim_a_source_hint": "August conversation",
                        "claim_b_source_hint": "December conversation",
                        "explanation": "Cannot own a BMW and daily-drive a Dodge Neon",
                        "confidence": "high",
                    }
                ]
            }
        )
        _mock_llm_for_validator(monkeypatch, conflict_json)

        validator = SemanticConflict(llm_config={"api_key": "test"})
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "semantic_conflict"
        assert violations[0].severity == "error"
        assert violations[0].metadata["claim_a"] == "owns a BMW"
        assert violations[0].metadata["explanation"] != ""
        assert violations[0].violation_id != ""

    def test_violation_id_deterministic(self, ctx, monkeypatch):
        """Same claims -> same violation_id."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")

        conflict = {"conflicts": [{"claim_a": "A", "claim_b": "B", "explanation": "x", "confidence": "high"}]}
        _mock_llm_for_validator(monkeypatch, json.dumps(conflict))

        validator = SemanticConflict(llm_config={"api_key": "test"})
        v1 = validator.validate([art], ctx)
        v2 = validator.validate([art], ctx)
        assert v1[0].violation_id == v2[0].violation_id

    def test_max_artifacts_respected(self, ctx, monkeypatch):
        """Only first max_artifacts are checked."""
        arts = [_make_artifact(f"m-{i}", "monthly", content=f"content {i}", layer_name="monthly") for i in range(5)]

        _, call_count = _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        validator = SemanticConflict(llm_config={"api_key": "test"}, max_artifacts=2)
        validator.validate(arts, ctx)
        assert call_count[0] == 2

    def test_llm_error_graceful_skip(self, ctx, monkeypatch):
        """LLM error -> skip artifact, no crash."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")

        _mock_llm_for_validator(monkeypatch, RuntimeError("LLM failed"))

        validator = SemanticConflict(llm_config={"api_key": "test"})
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_invalid_json_returns_empty(self, ctx, monkeypatch):
        """Invalid JSON from LLM -> no violations."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")

        _mock_llm_for_validator(monkeypatch, "not valid json at all")

        validator = SemanticConflict(llm_config={"api_key": "test"})
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_llm_trace_stored(self, build_dir, monkeypatch):
        """LLM call produces a trace artifact when store supports save_artifact."""
        from synix.build.artifacts import ArtifactStore

        writable_store = ArtifactStore(build_dir)
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        writable_store.save_artifact(art, "monthly", 2)

        _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        writable_ctx = ValidationContext(store=writable_store)
        validator = SemanticConflict(llm_config={"api_key": "test"})
        validator.validate([art], writable_ctx)

        traces = writable_store.list_artifacts("traces")
        assert len(traces) >= 1
        assert traces[0].artifact_type == "llm_trace"

    def test_no_llm_config_returns_empty(self, ctx):
        """No LLM config -> returns empty (can't create client)."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        validator = SemanticConflict()
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_instantiation(self):
        v = SemanticConflict()
        assert isinstance(v, SemanticConflict)
