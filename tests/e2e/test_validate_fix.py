"""E2E tests for the validate → fix → rebuild cycle."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.fixers import apply_fix, run_fixers
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import (
    Violation,
    ViolationQueue,
    compute_violation_id,
    run_validators,
)
from synix.core.models import Artifact, FixerDecl, Pipeline, ValidatorDecl

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


def _make_artifact(aid, content, atype="episode", layer_name="episodes", **meta):
    return Artifact(
        label=aid,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, **meta},
    )


# ---------------------------------------------------------------------------
# PII detection E2E
# ---------------------------------------------------------------------------


class TestPIIDetectionE2E:
    def test_pii_detected_in_episodes(self, store, provenance, build_dir):
        """Build artifacts with PII → validate detects them with provenance."""
        # Set up artifacts with provenance
        transcript = _make_artifact(
            "t-1",
            "User says their SSN is 123-45-6789",
            atype="transcript",
            layer_name="transcripts",
        )
        store.save_artifact(transcript, "transcripts", 0)

        episode = _make_artifact(
            "ep-1",
            "The user mentioned their SSN: 123-45-6789",
            atype="episode",
            layer_name="episodes",
        )
        store.save_artifact(episode, "episodes", 1)
        provenance.record("ep-1", parent_labels=["t-1"])

        pipeline = Pipeline("test")
        pipeline.add_validator(
            ValidatorDecl(
                name="pii",
                config={"layers": ["episodes"], "severity": "warning"},
            )
        )

        result = run_validators(pipeline, store, provenance)
        assert len(result.violations) >= 1
        ssn_viols = [v for v in result.violations if v.metadata.get("pattern") == "ssn"]
        assert len(ssn_viols) == 1
        assert ssn_viols[0].label == "ep-1"

        # Provenance trace should include ep-1 and t-1
        trace_ids = {s.label for s in ssn_viols[0].provenance_trace}
        assert "ep-1" in trace_ids


# ---------------------------------------------------------------------------
# Violation queue persistence
# ---------------------------------------------------------------------------


class TestViolationQueueE2E:
    def test_validate_persists_to_queue(self, store, provenance, build_dir):
        """Validate → violations saved to queue → load roundtrip."""
        art = _make_artifact("ep-1", "Email: user@example.com")
        store.save_artifact(art, "episodes", 1)

        pipeline = Pipeline("test")
        pipeline.add_validator(
            ValidatorDecl(
                name="pii",
                config={"layers": ["episodes"]},
            )
        )

        result = run_validators(pipeline, store, provenance)

        # Save to queue
        queue = ViolationQueue(build_dir=build_dir)
        for v in result.violations:
            queue.upsert(v)
        queue.save_state()

        # Verify files exist
        assert (build_dir / "violations_state.json").exists()
        assert (build_dir / "violations.jsonl").exists()

        # Reload and verify
        queue2 = ViolationQueue.load(build_dir)
        active = queue2.active()
        assert len(active) >= 1

    def test_violation_id_dedup(self, build_dir):
        """Upsert same violation twice → no duplicate in state."""
        queue = ViolationQueue(build_dir=build_dir)
        vid = compute_violation_id("pii", "ep-1", "email", "user@example.com")

        v1 = Violation(
            violation_type="pii",
            severity="warning",
            message="PII detected",
            label="ep-1",
            field="content",
            violation_id=vid,
            metadata={"artifact_id": "sha256:abc"},
        )
        queue.upsert(v1)
        queue.upsert(v1)
        queue.save_state()

        queue2 = ViolationQueue.load(build_dir)
        assert len(queue2.active()) == 1

    def test_ignore_flow(self, build_dir):
        """Ignore → validate again → not reported (same hash)."""
        vid = compute_violation_id("pii", "ep-1", "ssn", "123-45-6789")
        queue = ViolationQueue(build_dir=build_dir)

        v = Violation(
            violation_type="pii",
            severity="warning",
            message="PII",
            label="ep-1",
            field="content",
            violation_id=vid,
            metadata={"artifact_id": "sha256:abc"},
        )
        queue.upsert(v)
        queue.ignore(vid)
        queue.save_state()

        # Same content hash → still ignored
        assert queue.is_ignored(vid, "sha256:abc") is True

    def test_ignore_invalidation(self, build_dir):
        """Ignore → artifact rebuilt (new hash) → violation resurfaces."""
        vid = compute_violation_id("pii", "ep-1", "ssn", "123-45-6789")
        queue = ViolationQueue(build_dir=build_dir)

        v = Violation(
            violation_type="pii",
            severity="warning",
            message="PII",
            label="ep-1",
            field="content",
            violation_id=vid,
            metadata={"artifact_id": "sha256:old"},
        )
        queue.upsert(v)
        queue.ignore(vid)
        queue.save_state()

        # New content hash → ignore invalidated
        assert queue.is_ignored(vid, "sha256:new") is False


# ---------------------------------------------------------------------------
# Semantic conflict fix cycle
# ---------------------------------------------------------------------------


class TestSemanticFixCycleE2E:
    def _patch_llm(self, monkeypatch, conflict_client):
        """Patch LLMClient/LLMConfig so the validator creates conflict_client."""
        monkeypatch.setattr("synix.build.llm_client.LLMClient", lambda cfg: conflict_client)
        monkeypatch.setattr("synix.core.config.LLMConfig.from_dict", lambda d: None)

    def test_validate_fix_rebuild_cycle(self, store, provenance, build_dir, monkeypatch):
        """Full cycle: build → validate (conflict) → fix (accept) → verify artifact rewritten."""
        # Set up: monthly artifact with a contradiction
        monthly = _make_artifact(
            "monthly-dec",
            "Mark owns a BMW. Mark drives a Dodge Neon daily.",
            atype="monthly",
            layer_name="monthly",
        )
        store.save_artifact(monthly, "monthly", 2)

        # Episode sources
        ep1 = _make_artifact("ep-aug", "Mark talked about his BMW.", layer_name="episodes")
        ep2 = _make_artifact("ep-dec", "Mark drives a Dodge Neon.", layer_name="episodes")
        store.save_artifact(ep1, "episodes", 1)
        store.save_artifact(ep2, "episodes", 1)
        provenance.record("monthly-dec", parent_labels=["ep-aug", "ep-dec"])

        # Core depends on monthly
        core = _make_artifact(
            "core-1",
            "Core memory based on monthly summaries.",
            atype="core",
            layer_name="core",
        )
        store.save_artifact(core, "core", 3)
        provenance.record("core-1", parent_labels=["monthly-dec"])

        # Patch LLM to use mock conflict client
        self._patch_llm(monkeypatch, _make_mock_conflict_client())

        # Pipeline with semantic conflict validator + enrichment fixer
        pipeline = Pipeline("test")
        pipeline.add_validator(
            ValidatorDecl(
                name="semantic_conflict",
                config={
                    "layers": ["monthly"],
                    "llm_config": {"api_key": "test"},
                },
            )
        )
        pipeline.add_fixer(
            FixerDecl(
                name="semantic_enrichment",
                config={"max_context_episodes": 3},
            )
        )

        # Step 1: Validate — should find the contradiction
        result = run_validators(pipeline, store, provenance)
        assert len(result.violations) >= 1
        conflict_viols = [v for v in result.violations if v.violation_type == "semantic_conflict"]
        assert len(conflict_viols) >= 1

        # Step 2: Save violations to queue
        queue = ViolationQueue(build_dir=build_dir)
        for v in result.violations:
            queue.upsert(v)
        queue.save_state()
        assert (build_dir / "violations_state.json").exists()
        assert (build_dir / "violations.jsonl").exists()

        # Step 3: Run fixer
        mock_fix_client = _make_mock_fix_client_resolved()
        fix_result = run_fixers(
            result,
            pipeline,
            store,
            provenance,
            llm_client=mock_fix_client,
        )

        assert len(fix_result.actions) >= 1
        rewrite_actions = [a for a in fix_result.actions if a.action == "rewrite"]
        assert len(rewrite_actions) >= 1

        # Step 4: Apply fix (simulating user accept)
        for action in rewrite_actions:
            apply_fix(action, store, provenance)

            # Mark resolved in queue
            for v in result.violations:
                if v.label == action.label:
                    queue.resolve(v.violation_id, fix_action="rewrite")
            break

        queue.save_state()

        # Step 5: Verify artifact was rewritten
        reloaded = store.load_artifact("monthly-dec")
        assert reloaded is not None
        assert "previously owned" in reloaded.content

        # Step 6: Verify evidence provenance
        parents = provenance.get_parents("monthly-dec")
        assert "ep-aug" in parents
        assert "ep-dec" in parents

        # Step 7: Verify downstream invalidation
        assert "core-1" in fix_result.actions[0].downstream_invalidated

    def test_unresolved_accept_original(self, store, provenance, build_dir, monkeypatch):
        """Unresolved → user accepts original → violation resolved, no content change."""
        monthly = _make_artifact(
            "monthly-1",
            "Contradictory content here.",
            atype="monthly",
            layer_name="monthly",
        )
        store.save_artifact(monthly, "monthly", 2)
        original_hash = monthly.artifact_id

        # Patch LLM to use mock conflict client
        self._patch_llm(monkeypatch, _make_mock_conflict_client())

        pipeline = Pipeline("test")
        pipeline.add_validator(
            ValidatorDecl(
                name="semantic_conflict",
                config={
                    "layers": ["monthly"],
                    "llm_config": {"api_key": "test"},
                },
            )
        )
        pipeline.add_fixer(
            FixerDecl(
                name="semantic_enrichment",
            )
        )

        result = run_validators(pipeline, store, provenance)
        assert len(result.violations) >= 1

        # Run fixer — returns unresolved
        mock_client = _make_mock_fix_client_unresolved()
        fix_result = run_fixers(
            result,
            pipeline,
            store,
            provenance,
            llm_client=mock_client,
        )

        unresolved = [a for a in fix_result.actions if a.action == "unresolved"]
        assert len(unresolved) >= 1

        # User accepts original (mark resolved, no content change)
        queue = ViolationQueue(build_dir=build_dir)
        for v in result.violations:
            queue.upsert(v)
            queue.resolve(v.violation_id, fix_action="accept_original")
        queue.save_state()

        # Verify: artifact unchanged
        reloaded = store.load_artifact("monthly-1")
        assert reloaded.artifact_id == original_hash

        # Verify: resolved in queue
        queue2 = ViolationQueue.load(build_dir)
        assert len(queue2.active()) == 0


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJSONOutput:
    def test_validate_json_parseable(self, store, provenance, build_dir):
        """--json output includes reasoning and violation_ids."""
        art = _make_artifact("ep-1", "SSN: 123-45-6789")
        store.save_artifact(art, "episodes", 1)

        pipeline = Pipeline("test")
        pipeline.add_validator(
            ValidatorDecl(
                name="pii",
                config={"layers": ["episodes"]},
            )
        )

        result = run_validators(pipeline, store, provenance)
        out = result.to_dict()
        # Add violation_ids as the CLI does
        for i, v in enumerate(result.violations):
            out["violations"][i]["violation_id"] = v.violation_id

        parsed = json.loads(json.dumps(out))
        assert "violations" in parsed
        assert len(parsed["violations"]) >= 1
        assert "violation_id" in parsed["violations"][0]


# ---------------------------------------------------------------------------
# Mock LLM clients
# ---------------------------------------------------------------------------


def _make_mock_conflict_client():
    """Mock LLM client that always finds one conflict."""
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
    mock_response = MagicMock()
    mock_response.content = conflict_json
    client = MagicMock()
    client.complete = MagicMock(return_value=mock_response)
    return client


def _make_mock_fix_client_resolved():
    """Mock LLM client that resolves the contradiction."""
    resolved_json = json.dumps(
        {
            "status": "resolved",
            "content": "Mark previously owned a BMW. After an incident, he now drives a Dodge Neon daily.",
            "explanation": "Temporal resolution based on source context.",
        }
    )
    mock_response = MagicMock()
    mock_response.content = resolved_json
    client = MagicMock()
    client.complete = MagicMock(return_value=mock_response)
    return client


def _make_mock_fix_client_unresolved():
    """Mock LLM client that cannot resolve the contradiction."""
    unresolved_json = json.dumps(
        {
            "status": "unresolved",
            "content": "",
            "explanation": "Both claims appear without temporal markers. Cannot determine which is current.",
        }
    )
    mock_response = MagicMock()
    mock_response.content = unresolved_json
    client = MagicMock()
    client.complete = MagicMock(return_value=mock_response)
    return client
