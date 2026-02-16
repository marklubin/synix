"""E2E tests for the synix status command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.build.validators import Violation, ViolationQueue, compute_violation_id
from synix.core.models import Artifact

SYNIX_BIN = str(Path(sys.executable).parent / "synix")


def _run(*args: str, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [SYNIX_BIN, *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )


def _make_artifact(label, content, atype="episode", layer_name="episodes", **meta):
    return Artifact(
        label=label,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, **meta},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def built_dir(tmp_path):
    """Build dir with artifacts across two layers."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    store = ArtifactStore(build_dir)

    # Layer 0: sources
    for i in range(3):
        art = _make_artifact(
            f"src-{i}",
            f"Source content {i}",
            atype="transcript",
            layer_name="sources",
        )
        store.save_artifact(art, "sources", 0)

    # Layer 1: summaries
    for i in range(2):
        art = _make_artifact(
            f"sum-{i}",
            f"Summary content {i}",
            atype="episode",
            layer_name="summaries",
        )
        store.save_artifact(art, "summaries", 1)

    return build_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatusLayers:
    def test_shows_layer_counts(self, built_dir):
        """Status shows correct artifact count per layer."""
        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "sources" in result.stdout
        assert "summaries" in result.stdout
        # 3 source artifacts, 2 summary artifacts
        assert "3" in result.stdout
        assert "2" in result.stdout

    def test_shows_build_timestamps(self, built_dir):
        """Status shows non-empty timestamps for layers."""
        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        # Should have month abbreviation (e.g. "Feb") not just "-"
        # Count dashes that are used as placeholder timestamps
        lines = result.stdout.splitlines()
        layer_lines = [l for l in lines if "sources" in l or "summaries" in l]
        for line in layer_lines:
            # Each layer line should NOT end with just " - " as timestamp
            assert "│     -      │" not in line, f"Missing timestamp in: {line}"

    def test_empty_build_dir(self, tmp_path):
        """Status with empty build shows empty table."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        result = _run("status", "--build-dir", str(build_dir))
        assert result.returncode == 0
        assert "Build Status" in result.stdout

    def test_missing_build_dir(self, tmp_path):
        """Status with nonexistent build dir shows error."""
        result = _run("status", "--build-dir", str(tmp_path / "nonexistent"))
        assert result.returncode != 0 or "No build directory" in result.stdout


class TestStatusViolations:
    def test_no_violations_file(self, built_dir):
        """Status without violations_state.json shows no violation section."""
        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "Violations" not in result.stdout

    def test_active_violations_shown(self, built_dir):
        """Status shows active violations grouped by artifact."""
        queue = ViolationQueue(build_dir=built_dir)
        for i in range(3):
            vid = compute_violation_id("pii", "sum-0", f"pattern-{i}")
            queue.upsert(
                Violation(
                    violation_type="pii",
                    severity="error",
                    message=f"PII detected: pattern {i}",
                    label="sum-0",
                    field="content",
                    violation_id=vid,
                    metadata={"artifact_id": "sha256:abc"},
                )
            )
        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "3 active" in result.stdout
        assert "sum-0" in result.stdout
        assert "PII detected" in result.stdout

    def test_resolved_violations_counted(self, built_dir):
        """Status shows resolved count."""
        queue = ViolationQueue(build_dir=built_dir)
        vid = compute_violation_id("pii", "sum-0", "ssn")
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="warning",
                message="PII: SSN found",
                label="sum-0",
                field="content",
                violation_id=vid,
                metadata={"artifact_id": "sha256:abc"},
            )
        )
        queue.resolve(vid, fix_action="rewrite")
        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "1 resolved" in result.stdout
        # Resolved violations should NOT appear in the tree
        assert "PII: SSN found" not in result.stdout

    def test_mixed_statuses(self, built_dir):
        """Status shows active + resolved + ignored counts."""
        queue = ViolationQueue(build_dir=built_dir)

        # Active
        vid1 = compute_violation_id("pii", "sum-0", "email")
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="error",
                message="PII: email found",
                label="sum-0",
                field="content",
                violation_id=vid1,
                metadata={"artifact_id": "sha256:abc"},
            )
        )

        # Resolved
        vid2 = compute_violation_id("pii", "sum-1", "ssn")
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="warning",
                message="PII: SSN found",
                label="sum-1",
                field="content",
                violation_id=vid2,
                metadata={"artifact_id": "sha256:def"},
            )
        )
        queue.resolve(vid2, fix_action="rewrite")

        # Ignored
        vid3 = compute_violation_id("pii", "sum-0", "phone")
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="warning",
                message="PII: phone found",
                label="sum-0",
                field="content",
                violation_id=vid3,
                metadata={"artifact_id": "sha256:abc"},
            )
        )
        queue.ignore(vid3)

        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "1 active" in result.stdout
        assert "1 resolved" in result.stdout
        assert "1 ignored" in result.stdout
        # Only active violation in tree
        assert "PII: email found" in result.stdout
        assert "PII: SSN found" not in result.stdout

    def test_traces_excluded_from_layers(self, built_dir):
        """Trace artifacts should not appear in the layer table."""
        store = ArtifactStore(built_dir)
        trace = Artifact(
            label="trace-check-sum-0-abc123",
            artifact_type="llm_trace",
            content="{}",
            metadata={"trace_type": "citation_check", "layer_name": "traces"},
        )
        store.save_artifact(trace, "traces", 99)

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert (
            "traces" not in result.stdout.lower().split("violations")[0]
            if "Violations" in result.stdout
            else "traces" not in result.stdout
        )


# ---------------------------------------------------------------------------
# Stale artifact detection
# ---------------------------------------------------------------------------


class TestStatusStale:
    def _build_with_provenance(self, tmp_path):
        """Helper: build dir with parent→child provenance and matching input_ids."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        store = ArtifactStore(build_dir)
        provenance = ProvenanceTracker(build_dir)

        # Parent artifact
        parent = _make_artifact("strategy", "Strategy content v1", atype="episode", layer_name="episodes")
        store.save_artifact(parent, "episodes", 1)
        provenance.record("strategy", [])

        # Child artifact with parent's hash in input_ids
        parent_hash = store.get_artifact_id("strategy")
        child = Artifact(
            label="call-prep",
            artifact_type="core_memory",
            content="Call prep based on strategy",
            input_ids=[parent_hash],
            metadata={"layer_name": "core"},
        )
        store.save_artifact(child, "core", 2)
        provenance.record("call-prep", ["strategy"])

        return build_dir, store, provenance

    def test_stale_artifacts_detected(self, tmp_path):
        """Rewriting a parent makes the child stale."""
        build_dir, store, provenance = self._build_with_provenance(tmp_path)

        # Rewrite the parent with new content (changes its hash)
        new_parent = _make_artifact("strategy", "Strategy content v2 UPDATED", atype="episode", layer_name="episodes")
        store.save_artifact(new_parent, "episodes", 1)

        result = _run("status", "--build-dir", str(build_dir))
        assert result.returncode == 0
        assert "Stale Artifacts" in result.stdout
        assert "call-prep" in result.stdout
        assert "parent changed: strategy" in result.stdout

    def test_no_stale_when_fresh(self, tmp_path):
        """Consistent hashes produce no stale section."""
        build_dir, _, _ = self._build_with_provenance(tmp_path)

        result = _run("status", "--build-dir", str(build_dir))
        assert result.returncode == 0
        assert "Stale" not in result.stdout


# ---------------------------------------------------------------------------
# Resolved violation detail (--resolved flag)
# ---------------------------------------------------------------------------


class TestStatusResolved:
    def test_resolved_flag_shows_details(self, built_dir):
        """--resolved shows fix_action, message, and timestamp."""
        queue = ViolationQueue(build_dir=built_dir)
        vid = compute_violation_id("citation", "sum-0", "ungrounded")
        queue.upsert(
            Violation(
                violation_type="citation",
                severity="warning",
                message="Ungrounded claim: Smart Insights module",
                label="sum-0",
                field="content",
                violation_id=vid,
                metadata={"artifact_id": "sha256:abc"},
            )
        )
        queue.resolve(vid, fix_action="rewrite")
        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir), "--resolved")
        assert result.returncode == 0
        assert "Resolved Violations" in result.stdout
        assert "rewrite" in result.stdout
        assert "Ungrounded claim" in result.stdout

    def test_resolved_flag_absent_hides_details(self, built_dir):
        """Without --resolved, resolved details are hidden (only count shown)."""
        queue = ViolationQueue(build_dir=built_dir)
        vid = compute_violation_id("citation", "sum-0", "claim")
        queue.upsert(
            Violation(
                violation_type="citation",
                severity="warning",
                message="Ungrounded claim: hidden detail",
                label="sum-0",
                field="content",
                violation_id=vid,
                metadata={"artifact_id": "sha256:abc"},
            )
        )
        queue.resolve(vid, fix_action="rewrite")
        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "1 resolved" in result.stdout
        assert "Resolved Violations" not in result.stdout


# ---------------------------------------------------------------------------
# Next-step guidance
# ---------------------------------------------------------------------------


class TestStatusNextSteps:
    def test_next_steps_active_violations(self, built_dir):
        """Active violations → guidance says fix."""
        queue = ViolationQueue(build_dir=built_dir)
        vid = compute_violation_id("pii", "sum-0", "email")
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="error",
                message="PII: email found",
                label="sum-0",
                field="content",
                violation_id=vid,
                metadata={"artifact_id": "sha256:abc"},
            )
        )
        queue.save_state()

        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "synix fix" in result.stdout

    def test_next_steps_stale_only(self, tmp_path):
        """Stale artifacts only → guidance says build."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        store = ArtifactStore(build_dir)
        provenance = ProvenanceTracker(build_dir)

        # Parent
        parent = _make_artifact("monthly-dec", "December rollup", atype="rollup", layer_name="rollups")
        store.save_artifact(parent, "rollups", 1)
        provenance.record("monthly-dec", [])

        # Child with old parent hash
        child = Artifact(
            label="core-1",
            artifact_type="core_memory",
            content="Core synthesis",
            input_ids=["sha256:old_hash_that_doesnt_match"],
            metadata={"layer_name": "core"},
        )
        store.save_artifact(child, "core", 2)
        provenance.record("core-1", ["monthly-dec"])

        result = _run("status", "--build-dir", str(build_dir))
        assert result.returncode == 0
        assert "synix build" in result.stdout
        assert "stale" in result.stdout.lower()

    def test_next_steps_all_clean(self, built_dir):
        """No violations + no stale → build is clean."""
        result = _run("status", "--build-dir", str(built_dir))
        assert result.returncode == 0
        assert "Build is clean" in result.stdout


# ---------------------------------------------------------------------------
# fix_action persistence
# ---------------------------------------------------------------------------


class TestFixActionPersistence:
    def test_fix_action_persisted(self, tmp_path):
        """resolve() persists fix_action and resolved_at in state."""

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        queue = ViolationQueue(build_dir=build_dir)
        vid = compute_violation_id("citation", "art-1", "claim-x")
        queue.upsert(
            Violation(
                violation_type="citation",
                severity="warning",
                message="Bad claim",
                label="art-1",
                field="content",
                violation_id=vid,
                metadata={"artifact_id": "sha256:xyz"},
            )
        )
        queue.resolve(vid, fix_action="rewrite")
        queue.save_state()

        # Reload from disk
        reloaded = ViolationQueue.load(build_dir)
        state = reloaded._state[vid]
        assert state["status"] == "resolved"
        assert state["fix_action"] == "rewrite"
        assert "resolved_at" in state
        # resolved_at should be a valid ISO timestamp
        from datetime import datetime

        datetime.fromisoformat(state["resolved_at"])
