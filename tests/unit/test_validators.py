"""Unit tests for the pluggable validator framework."""

from __future__ import annotations

import json

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import (
    BaseValidator,
    MutualExclusionValidator,
    PIIValidator,
    ProvenanceStep,
    RequiredFieldValidator,
    SemanticConflictValidator,
    ValidationContext,
    ValidationResult,
    Violation,
    ViolationQueue,
    _gather_artifacts,
    _parse_conflict_response,
    compute_violation_id,
    get_validator,
    mutual_exclusion_violation,
    register_validator,
    required_field_violation,
    run_validators,
)
from synix.core.models import Artifact, Pipeline, ValidatorDecl

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
        artifact_id=aid,
        artifact_type=atype,
        content=content,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestViolation:
    def test_basic_fields(self):
        v = Violation(
            violation_type="mutual_exclusion",
            severity="error",
            message="test message",
            artifact_id="art-1",
            field="customer_id",
        )
        assert v.violation_type == "mutual_exclusion"
        assert v.severity == "error"
        assert v.artifact_id == "art-1"
        assert v.field == "customer_id"
        assert v.provenance_trace == []
        assert v.metadata == {}

    def test_with_provenance_trace(self):
        steps = [
            ProvenanceStep(artifact_id="a", layer="merge", field_value="acme"),
            ProvenanceStep(artifact_id="b", layer="episode", field_value="acme"),
        ]
        v = Violation(
            violation_type="test",
            severity="warning",
            message="msg",
            artifact_id="a",
            field="f",
            provenance_trace=steps,
        )
        assert len(v.provenance_trace) == 2
        assert v.provenance_trace[0].field_value == "acme"


class TestProvenanceStep:
    def test_defaults(self):
        s = ProvenanceStep(artifact_id="x", layer="episodes")
        assert s.field_value is None

    def test_with_value(self):
        s = ProvenanceStep(artifact_id="x", layer="episodes", field_value="acme")
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
                    violation_type="test", severity="error",
                    message="msg", artifact_id="a", field="f",
                )
            ],
            validators_run=["test"],
        )
        assert r.passed is False

    def test_warning_only_passes(self):
        r = ValidationResult(
            violations=[
                Violation(
                    violation_type="test", severity="warning",
                    message="msg", artifact_id="a", field="f",
                )
            ],
            validators_run=["test"],
        )
        assert r.passed is True

    def test_to_dict(self):
        r = ValidationResult(
            violations=[
                Violation(
                    violation_type="mutual_exclusion", severity="error",
                    message="test", artifact_id="a", field="customer_id",
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
        assert d["violations"][0]["provenance_trace"][0]["artifact_id"] == "a"
        assert "mutual_exclusion" in d["validators_run"]


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

class TestFactoryFunctions:
    def test_mutual_exclusion_violation(self):
        v = mutual_exclusion_violation("art-1", "customer_id", ["acme", "globex"])
        assert v.violation_type == "mutual_exclusion"
        assert v.severity == "error"
        assert v.artifact_id == "art-1"
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
        assert v.artifact_id == "art-2"
        assert v.field == "customer_id"
        assert "customer_id" in v.message

    def test_required_field_custom_message(self):
        v = required_field_violation("a", "f", message="custom")
        assert v.message == "custom"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_mutual_exclusion(self):
        v = get_validator("mutual_exclusion")
        assert isinstance(v, MutualExclusionValidator)

    def test_get_required_field(self):
        v = get_validator("required_field")
        assert isinstance(v, RequiredFieldValidator)

    def test_unknown_validator_raises(self):
        with pytest.raises(ValueError, match="Unknown validator"):
            get_validator("nonexistent_validator")

    def test_register_custom_validator(self):
        @register_validator("test_custom")
        class CustomValidator(BaseValidator):
            def validate(self, artifacts, ctx):
                return []

        v = get_validator("test_custom")
        assert isinstance(v, CustomValidator)
        assert v.validate([], None) == []


# ---------------------------------------------------------------------------
# ValidationContext.trace_field_origin tests
# ---------------------------------------------------------------------------

class TestTraceFieldOrigin:
    def test_single_artifact_no_parents(self, store, provenance, ctx):
        art = _make_artifact("t-1", "transcript", customer_id="acme",
                             layer_name="transcripts")
        store.save_artifact(art, "transcripts", 0)

        steps = ctx.trace_field_origin("t-1", "customer_id")
        assert len(steps) == 1
        assert steps[0].artifact_id == "t-1"
        assert steps[0].field_value == "acme"
        assert steps[0].layer == "transcripts"

    def test_two_level_chain(self, store, provenance, ctx):
        t = _make_artifact("t-1", "transcript", customer_id="acme",
                           layer_name="transcripts")
        store.save_artifact(t, "transcripts", 0)

        ep = _make_artifact("ep-1", "episode", customer_id="acme",
                            layer_name="episodes")
        store.save_artifact(ep, "episodes", 1)

        provenance.record("ep-1", parent_ids=["t-1"])

        steps = ctx.trace_field_origin("ep-1", "customer_id")
        assert len(steps) == 2
        ids = [s.artifact_id for s in steps]
        assert "ep-1" in ids
        assert "t-1" in ids

    def test_missing_field_returns_none_value(self, store, provenance, ctx):
        art = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store.save_artifact(art, "transcripts", 0)

        steps = ctx.trace_field_origin("t-1", "customer_id")
        assert len(steps) == 1
        assert steps[0].field_value is None

    def test_branching_provenance(self, store, provenance, ctx):
        """Merge artifact with two parents from different customers."""
        t1 = _make_artifact("t-1", "transcript", customer_id="acme",
                            layer_name="transcripts")
        t2 = _make_artifact("t-2", "transcript", customer_id="globex",
                            layer_name="transcripts")
        store.save_artifact(t1, "transcripts", 0)
        store.save_artifact(t2, "transcripts", 0)

        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        provenance.record("ep-1", parent_ids=["t-1"])
        provenance.record("ep-2", parent_ids=["t-2"])

        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)
        provenance.record("merge-ep-1", parent_ids=["ep-1", "ep-2"])

        steps = ctx.trace_field_origin("merge-ep-1", "customer_id")
        ids = {s.artifact_id for s in steps}
        assert "merge-ep-1" in ids
        assert "ep-1" in ids
        assert "ep-2" in ids
        assert "t-1" in ids
        assert "t-2" in ids


# ---------------------------------------------------------------------------
# MutualExclusionValidator tests
# ---------------------------------------------------------------------------

class TestMutualExclusionValidator:
    def test_no_violations_single_customer(self, store, provenance, ctx):
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="acme",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)
        provenance.record("merge-ep-1", parent_ids=["ep-1", "ep-2"])

        validator = get_validator("mutual_exclusion")
        validator._field_name = "customer_id"
        violations = validator.validate([merge], ctx)
        assert violations == []

    def test_violation_multiple_customers(self, store, provenance, ctx):
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        merge = _make_artifact("merge-ep-1", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)
        provenance.record("merge-ep-1", parent_ids=["ep-1", "ep-2"])

        validator = get_validator("mutual_exclusion")
        validator._field_name = "customer_id"
        violations = validator.validate([merge], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "mutual_exclusion"
        assert violations[0].artifact_id == "merge-ep-1"
        assert "acme" in violations[0].metadata["conflicting_values"]
        assert "globex" in violations[0].metadata["conflicting_values"]

    def test_source_field_metadata_detected(self, store, provenance, ctx):
        """source_customer_ids in merge artifact metadata also triggers."""
        merge = _make_artifact(
            "merge-ep-2", "merge", layer_name="merge",
            source_customer_ids=["acme", "globex"],
        )
        store.save_artifact(merge, "merge", 2)

        validator = get_validator("mutual_exclusion")
        validator._field_name = "customer_id"
        violations = validator.validate([merge], ctx)
        assert len(violations) == 1

    def test_no_parents_no_violation(self, store, provenance, ctx):
        """Artifact with no parents and no source metadata: no violation."""
        merge = _make_artifact("merge-ep-3", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)

        validator = get_validator("mutual_exclusion")
        validator._field_name = "customer_id"
        violations = validator.validate([merge], ctx)
        assert violations == []

    def test_multiple_merge_artifacts(self, store, provenance, ctx):
        """Only violating merge artifacts are flagged."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex",
                             layer_name="episodes")
        ep3 = _make_artifact("ep-3", "episode", customer_id="acme",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)
        store.save_artifact(ep3, "episodes", 1)

        # merge-1: cross-customer (violation)
        m1 = _make_artifact("merge-1", "merge", layer_name="merge")
        store.save_artifact(m1, "merge", 2)
        provenance.record("merge-1", parent_ids=["ep-1", "ep-2"])

        # merge-2: single customer (no violation)
        m2 = _make_artifact("merge-2", "merge", layer_name="merge")
        store.save_artifact(m2, "merge", 2)
        provenance.record("merge-2", parent_ids=["ep-1", "ep-3"])

        validator = get_validator("mutual_exclusion")
        validator._field_name = "customer_id"
        violations = validator.validate([m1, m2], ctx)
        assert len(violations) == 1
        assert violations[0].artifact_id == "merge-1"


# ---------------------------------------------------------------------------
# RequiredFieldValidator tests
# ---------------------------------------------------------------------------

class TestRequiredFieldValidator:
    def test_no_violations_field_present(self, store, provenance, ctx):
        art = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        store.save_artifact(art, "episodes", 1)

        validator = get_validator("required_field")
        validator._field_name = "customer_id"
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_violation_field_missing(self, store, provenance, ctx):
        art = _make_artifact("ep-1", "episode", layer_name="episodes")
        store.save_artifact(art, "episodes", 1)

        validator = get_validator("required_field")
        validator._field_name = "customer_id"
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "required_field"
        assert violations[0].artifact_id == "ep-1"

    def test_violation_field_empty_string(self, store, provenance, ctx):
        art = _make_artifact("ep-1", "episode", customer_id="",
                             layer_name="episodes")
        store.save_artifact(art, "episodes", 1)

        validator = get_validator("required_field")
        validator._field_name = "customer_id"
        violations = validator.validate([art], ctx)
        assert len(violations) == 1

    def test_violation_field_whitespace(self, store, provenance, ctx):
        art = _make_artifact("ep-1", "episode", customer_id="  ",
                             layer_name="episodes")
        store.save_artifact(art, "episodes", 1)

        validator = get_validator("required_field")
        validator._field_name = "customer_id"
        violations = validator.validate([art], ctx)
        assert len(violations) == 1

    def test_multiple_artifacts_mixed(self, store, provenance, ctx):
        a1 = _make_artifact("ep-1", "episode", customer_id="acme",
                            layer_name="episodes")
        a2 = _make_artifact("ep-2", "episode", layer_name="episodes")
        a3 = _make_artifact("ep-3", "episode", customer_id="globex",
                            layer_name="episodes")
        store.save_artifact(a1, "episodes", 1)
        store.save_artifact(a2, "episodes", 1)
        store.save_artifact(a3, "episodes", 1)

        validator = get_validator("required_field")
        validator._field_name = "customer_id"
        violations = validator.validate([a1, a2, a3], ctx)
        assert len(violations) == 1
        assert violations[0].artifact_id == "ep-2"


# ---------------------------------------------------------------------------
# _gather_artifacts tests
# ---------------------------------------------------------------------------

class TestGatherArtifacts:
    def test_filter_by_layers(self, store):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store.save_artifact(a1, "episodes", 1)
        store.save_artifact(a2, "transcripts", 0)

        result = _gather_artifacts(store, {"layers": ["episodes"]})
        assert len(result) == 1
        assert result[0].artifact_id == "ep-1"

    def test_filter_by_scope_prefix(self, store):
        a1 = _make_artifact("merge-1", "merge", layer_name="merge")
        a2 = _make_artifact("ep-1", "episode", layer_name="episodes")
        store.save_artifact(a1, "merge", 2)
        store.save_artifact(a2, "episodes", 1)

        result = _gather_artifacts(store, {"scope": "merge"})
        assert len(result) == 1
        assert result[0].artifact_id == "merge-1"

    def test_no_filter_returns_all(self, store):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        store.save_artifact(a1, "episodes", 1)
        store.save_artifact(a2, "transcripts", 0)

        result = _gather_artifacts(store, {})
        assert len(result) == 2

    def test_filter_multiple_layers(self, store):
        a1 = _make_artifact("ep-1", "episode", layer_name="episodes")
        a2 = _make_artifact("t-1", "transcript", layer_name="transcripts")
        a3 = _make_artifact("m-1", "merge", layer_name="merge")
        store.save_artifact(a1, "episodes", 1)
        store.save_artifact(a2, "transcripts", 0)
        store.save_artifact(a3, "merge", 2)

        result = _gather_artifacts(store, {"layers": ["episodes", "merge"]})
        ids = {a.artifact_id for a in result}
        assert ids == {"ep-1", "m-1"}


# ---------------------------------------------------------------------------
# run_validators integration tests
# ---------------------------------------------------------------------------

class TestRunValidators:
    def test_no_validators_empty_result(self, store, provenance):
        pipeline = Pipeline("test")
        result = run_validators(pipeline, store, provenance)
        assert result.passed is True
        assert result.validators_run == []
        assert result.violations == []

    def test_mutual_exclusion_end_to_end(self, store, provenance):
        """Full integration: pipeline declares mutual_exclusion validator."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        merge = _make_artifact("merge-1", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)
        provenance.record("merge-1", parent_ids=["ep-1", "ep-2"])

        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(
            name="mutual_exclusion",
            config={"field": "customer_id", "scope": "merge"},
        ))

        result = run_validators(pipeline, store, provenance)
        assert result.passed is False
        assert "mutual_exclusion" in result.validators_run
        assert len(result.violations) == 1
        assert result.violations[0].artifact_id == "merge-1"

    def test_required_field_end_to_end(self, store, provenance):
        """Full integration: pipeline declares required_field validator."""
        a1 = _make_artifact("ep-1", "episode", customer_id="acme",
                            layer_name="episodes")
        a2 = _make_artifact("ep-2", "episode", layer_name="episodes")
        store.save_artifact(a1, "episodes", 1)
        store.save_artifact(a2, "episodes", 1)

        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(
            name="required_field",
            config={"field": "customer_id", "layers": ["episodes"]},
        ))

        result = run_validators(pipeline, store, provenance)
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].artifact_id == "ep-2"

    def test_multiple_validators(self, store, provenance):
        """Both validators run; results aggregated."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(
            name="required_field",
            config={"field": "customer_id", "layers": ["episodes"]},
        ))
        pipeline.add_validator(ValidatorDecl(
            name="mutual_exclusion",
            config={"field": "customer_id", "scope": "merge"},
        ))

        result = run_validators(pipeline, store, provenance)
        assert len(result.validators_run) == 2
        # Only required_field should have a violation (no merge artifacts)
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "required_field"

    def test_provenance_auto_resolved(self, store, provenance):
        """Violations get provenance traces auto-resolved."""
        t1 = _make_artifact("t-1", "transcript", customer_id="acme",
                            layer_name="transcripts")
        store.save_artifact(t1, "transcripts", 0)

        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="globex",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)
        provenance.record("ep-1", parent_ids=["t-1"])

        merge = _make_artifact("merge-1", "merge", layer_name="merge")
        store.save_artifact(merge, "merge", 2)
        provenance.record("merge-1", parent_ids=["ep-1", "ep-2"])

        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(
            name="mutual_exclusion",
            config={"field": "customer_id", "scope": "merge"},
        ))

        result = run_validators(pipeline, store, provenance)
        assert len(result.violations) == 1
        trace = result.violations[0].provenance_trace
        assert len(trace) >= 2  # at least merge-1 and parents
        trace_ids = {s.artifact_id for s in trace}
        assert "merge-1" in trace_ids
        assert "ep-1" in trace_ids

    def test_all_pass(self, store, provenance):
        """No violations when all artifacts are valid."""
        ep1 = _make_artifact("ep-1", "episode", customer_id="acme",
                             layer_name="episodes")
        ep2 = _make_artifact("ep-2", "episode", customer_id="acme",
                             layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)

        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(
            name="required_field",
            config={"field": "customer_id", "layers": ["episodes"]},
        ))

        result = run_validators(pipeline, store, provenance)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_unknown_validator_raises(self, store, provenance):
        pipeline = Pipeline("test")
        pipeline.add_validator(ValidatorDecl(name="nonexistent"))

        with pytest.raises(ValueError, match="Unknown validator"):
            run_validators(pipeline, store, provenance)


# ---------------------------------------------------------------------------
# ValidatorDecl model tests
# ---------------------------------------------------------------------------

class TestValidatorDecl:
    def test_basic(self):
        decl = ValidatorDecl(name="mutual_exclusion")
        assert decl.name == "mutual_exclusion"
        assert decl.config == {}

    def test_with_config(self):
        decl = ValidatorDecl(
            name="required_field",
            config={"field": "customer_id", "layers": ["episodes"]},
        )
        assert decl.config["field"] == "customer_id"

    def test_pipeline_add_validator(self):
        p = Pipeline("test")
        assert p.validators == []
        p.add_validator(ValidatorDecl(name="mutual_exclusion"))
        assert len(p.validators) == 1
        assert p.validators[0].name == "mutual_exclusion"

    def test_pipeline_multiple_validators(self):
        p = Pipeline("test")
        p.add_validator(ValidatorDecl(name="mutual_exclusion"))
        p.add_validator(ValidatorDecl(name="required_field"))
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
# PIIValidator tests
# ---------------------------------------------------------------------------

class TestPIIValidator:
    def test_detects_credit_card(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="My card is 4111-1111-1111-1111 ok?")
        violations = PIIValidator().validate([art], ctx)
        cc_violations = [v for v in violations if v.metadata["pattern"] == "credit_card"]
        assert len(cc_violations) == 1
        assert cc_violations[0].violation_type == "pii"
        assert "4111" in cc_violations[0].metadata["redacted_value"]
        assert "****" in cc_violations[0].metadata["redacted_value"]

    def test_detects_ssn(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="SSN: 123-45-6789")
        violations = PIIValidator().validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "ssn"
        assert "6789" in violations[0].metadata["redacted_value"]
        assert violations[0].metadata["redacted_value"].startswith("***-**-")

    def test_detects_email(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="Email me at user@example.com")
        violations = PIIValidator().validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "email"
        assert violations[0].metadata["redacted_value"] == "us***@example.com"

    def test_detects_phone(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="Call 555-867-5309")
        violations = PIIValidator().validate([art], ctx)
        assert len(violations) >= 1
        phone_violations = [v for v in violations if v.metadata["pattern"] == "phone"]
        assert len(phone_violations) >= 1
        assert "5309" in phone_violations[0].metadata["redacted_value"]

    def test_clean_text_no_violations(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="This is clean text with no PII")
        violations = PIIValidator().validate([art], ctx)
        assert violations == []

    def test_multiple_pii_in_one_artifact(self, store, provenance, ctx):
        art = _make_artifact(
            "a-1",
            content="Card: 4111-1111-1111-1111 and SSN: 123-45-6789",
        )
        violations = PIIValidator().validate([art], ctx)
        patterns = {v.metadata["pattern"] for v in violations}
        assert "credit_card" in patterns
        assert "ssn" in patterns

    def test_patterns_config_filter(self, store, provenance, ctx):
        art = _make_artifact(
            "a-1",
            content="Card: 4111-1111-1111-1111 and SSN: 123-45-6789",
        )
        validator = PIIValidator()
        validator._config = {"patterns": ["ssn"]}
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].metadata["pattern"] == "ssn"

    def test_violation_has_violation_id(self, store, provenance, ctx):
        art = _make_artifact("a-1", content="SSN: 123-45-6789")
        violations = PIIValidator().validate([art], ctx)
        assert violations[0].violation_id != ""

    def test_registered_in_registry(self):
        v = get_validator("pii")
        assert isinstance(v, PIIValidator)


# ---------------------------------------------------------------------------
# ViolationQueue tests
# ---------------------------------------------------------------------------

class TestViolationQueue:
    def test_save_load_roundtrip(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        q.save_state()

        q2 = ViolationQueue.load(build_dir)
        assert len(q2.active()) == 1
        assert q2.active()[0]["violation_id"] == "vid-1"

    def test_upsert_new(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        assert len(q.active()) == 1

    def test_upsert_existing_stays_active(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v1 = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v1)
        # Upsert again with same hash
        v2 = Violation(
            violation_type="pii", severity="warning", message="test updated",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v2)
        assert len(q.active()) == 1

    def test_ignore(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert len(q.active()) == 0

    def test_is_ignored(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert q.is_ignored("vid-1", "hash1") is True

    def test_is_ignored_invalidated_on_hash_change(self, build_dir):
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
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
            violation_type="pii", severity="warning", message="test1",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        v2 = Violation(
            violation_type="pii", severity="warning", message="test2",
            artifact_id="a-2", field="content", violation_id="vid-2",
            metadata={"content_hash": "hash2"},
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
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        q.resolve("vid-1", fix_action="redacted")
        assert len(q.active()) == 0

    def test_append_log_writes_jsonl(self, build_dir):
        import json
        q = ViolationQueue(build_dir=build_dir)
        v = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
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
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash1"},
        )
        q.upsert(v)
        q.ignore("vid-1")
        assert len(q.active()) == 0

        # Upsert with different hash should reactivate
        v2 = Violation(
            violation_type="pii", severity="warning", message="test",
            artifact_id="a-1", field="content", violation_id="vid-1",
            metadata={"content_hash": "hash2"},
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

    def test_invalid_json(self):
        resp = "this is not json at all"
        result = _parse_conflict_response(resp)
        assert result == []

    def test_missing_source_hints_defaulted(self):
        resp = '{"conflicts": [{"claim_a": "A", "claim_b": "B"}]}'
        result = _parse_conflict_response(resp)
        assert result[0]["claim_a_source_hint"] == ""
        assert result[0]["claim_b_source_hint"] == ""

    def test_conflicts_not_a_list(self):
        resp = '{"conflicts": "not a list"}'
        result = _parse_conflict_response(resp)
        assert result == []


# ---------------------------------------------------------------------------
# SemanticConflictValidator tests
# ---------------------------------------------------------------------------

def _mock_llm_for_validator(monkeypatch, response_content):
    """Set up monkeypatches so SemanticConflictValidator can create an LLM client.

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
    monkeypatch.setattr(
        "synix.build.llm_client.LLMClient", lambda cfg: mock_client
    )
    monkeypatch.setattr(
        "synix.core.config.LLMConfig.from_dict", lambda d: None
    )
    return mock_client, call_count


class TestSemanticConflictValidator:
    def test_no_conflicts(self, store, provenance, ctx, monkeypatch):
        """LLM returns no conflicts -> no violations."""
        art = _make_artifact("monthly-1", "monthly",
                             content="Mark likes Python.",
                             layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        validator = get_validator("semantic_conflict")
        validator._config = {"layers": ["monthly"], "llm_config": {"api_key": "test"}}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_conflict_found(self, store, provenance, ctx, monkeypatch):
        """LLM finds a conflict -> violation with metadata."""
        art = _make_artifact("monthly-1", "monthly",
                             content="Mark owns a BMW. Mark drives a Dodge Neon daily.",
                             layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        conflict_json = json.dumps({
            "conflicts": [{
                "claim_a": "owns a BMW",
                "claim_b": "drives a Dodge Neon daily",
                "claim_a_source_hint": "August conversation",
                "claim_b_source_hint": "December conversation",
                "explanation": "Cannot own a BMW and daily-drive a Dodge Neon",
                "confidence": "high",
            }]
        })
        _mock_llm_for_validator(monkeypatch, conflict_json)

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}}
        violations = validator.validate([art], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "semantic_conflict"
        assert violations[0].severity == "error"
        assert violations[0].metadata["claim_a"] == "owns a BMW"
        assert violations[0].metadata["explanation"] != ""
        assert violations[0].violation_id != ""

    def test_violation_id_deterministic(self, store, provenance, ctx, monkeypatch):
        """Same claims -> same violation_id."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        conflict = {"conflicts": [{"claim_a": "A", "claim_b": "B",
                                    "explanation": "x", "confidence": "high"}]}
        _mock_llm_for_validator(monkeypatch, json.dumps(conflict))

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}}
        v1 = validator.validate([art], ctx)
        v2 = validator.validate([art], ctx)
        assert v1[0].violation_id == v2[0].violation_id

    def test_max_artifacts_respected(self, store, provenance, ctx, monkeypatch):
        """Only first max_artifacts are checked."""
        arts = []
        for i in range(5):
            a = _make_artifact(f"m-{i}", "monthly", content=f"content {i}",
                               layer_name="monthly")
            store.save_artifact(a, "monthly", 2)
            arts.append(a)

        _, call_count = _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}, "max_artifacts": 2}
        validator.validate(arts, ctx)
        assert call_count[0] == 2

    def test_llm_error_graceful_skip(self, store, provenance, ctx, monkeypatch):
        """LLM error -> skip artifact, no crash."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        _mock_llm_for_validator(monkeypatch, RuntimeError("LLM failed"))

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_invalid_json_returns_empty(self, store, provenance, ctx, monkeypatch):
        """Invalid JSON from LLM -> no violations."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        _mock_llm_for_validator(monkeypatch, "not valid json at all")

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_llm_trace_stored(self, store, provenance, ctx, monkeypatch):
        """LLM call produces a trace artifact in layer99-traces."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        store.save_artifact(art, "monthly", 2)

        _mock_llm_for_validator(monkeypatch, '{"conflicts": []}')

        validator = get_validator("semantic_conflict")
        validator._config = {"llm_config": {"api_key": "test"}}
        validator.validate([art], ctx)

        traces = store.list_artifacts("traces")
        assert len(traces) >= 1
        assert traces[0].artifact_type == "llm_trace"

    def test_no_llm_config_returns_empty(self, store, provenance, ctx):
        """No LLM config -> returns empty (can't create client)."""
        art = _make_artifact("m-1", "monthly", content="test", layer_name="monthly")
        validator = get_validator("semantic_conflict")
        validator._config = {}
        violations = validator.validate([art], ctx)
        assert violations == []

    def test_registered_in_registry(self):
        v = get_validator("semantic_conflict")
        assert isinstance(v, SemanticConflictValidator)
