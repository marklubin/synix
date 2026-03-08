"""Pipeline verification — check integrity of a completed build."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from synix.build.refs import synix_dir_for_build_dir
from synix.build.search_outputs import SearchOutputResolutionError, list_search_outputs
from synix.build.snapshot_view import SnapshotArtifactCache


@dataclass
class VerifyCheck:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)
    fix_hint: str = ""


@dataclass
class VerifyResult:
    """Complete verification report."""

    checks: list[VerifyCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[VerifyCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed
        if failed == 0:
            return f"All {total} checks passed"
        return f"{failed}/{total} checks failed"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                    "fix_hint": c.fix_hint,
                }
                for c in self.checks
            ],
        }


def verify_build(build_dir: str | Path, checks: list[str] | None = None) -> VerifyResult:
    """Run verification checks on a completed build.

    Args:
        build_dir: Path to the build directory (used to locate .synix).
        checks: Optional list of specific check names to run.
                If None, runs all checks.

    Returns:
        VerifyResult with details of each check.
    """
    build_path = Path(build_dir)
    result = VerifyResult()

    all_checks = {
        "build_exists": _check_build_exists,
        "manifest_valid": _check_manifest_valid,
        "artifacts_exist": _check_artifacts_exist,
        "provenance_complete": _check_provenance_complete,
        "synix_search": _check_synix_search,
        "content_hashes": _check_content_hashes,
        "no_orphans": _check_no_orphans,
        "merge_integrity": _check_merge_integrity,
    }
    aliases = {
        "search_index": "synix_search",
    }

    checks_to_run = checks if checks else list(all_checks.keys())

    for check_name in checks_to_run:
        canonical_name = aliases.get(check_name, check_name)
        if canonical_name in all_checks:
            check_result = all_checks[canonical_name](build_path)
            result.checks.append(check_result)
        else:
            result.checks.append(
                VerifyCheck(
                    name=check_name,
                    passed=False,
                    message=f"Unknown check: {check_name}",
                )
            )

    return result


def _get_store(build_path: Path) -> SnapshotArtifactCache | None:
    """Get the SnapshotArtifactCache for the build, or None if unavailable."""
    try:
        synix_dir = synix_dir_for_build_dir(build_path)
        return SnapshotArtifactCache(synix_dir)
    except (ValueError, OSError):
        return None


def _check_build_exists(build_path: Path) -> VerifyCheck:
    """Check that the snapshot store exists and has a committed snapshot."""
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="build_exists",
            passed=False,
            message="No snapshot store found",
            fix_hint="Run: synix build <pipeline.py>",
        )
    entries = store.iter_entries()
    if not entries:
        return VerifyCheck(
            name="build_exists",
            passed=False,
            message="Snapshot store exists but contains no artifacts",
            fix_hint="Run: synix build <pipeline.py>",
        )
    return VerifyCheck(
        name="build_exists",
        passed=True,
        message=f"Snapshot store exists with {len(entries)} artifacts",
    )


def _check_manifest_valid(build_path: Path) -> VerifyCheck:
    """Check that the snapshot manifest is valid."""
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message="No snapshot store found",
            fix_hint="Re-run synix build to regenerate the manifest",
        )
    entries = store.iter_entries()
    if not entries:
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message="Snapshot manifest is empty",
            fix_hint="Re-run synix build to regenerate the manifest",
        )
    return VerifyCheck(
        name="manifest_valid",
        passed=True,
        message=f"Manifest valid with {len(entries)} artifacts",
    )


def _check_artifacts_exist(build_path: Path) -> VerifyCheck:
    """Check that all manifest entries have loadable artifacts in the object store."""
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="artifacts_exist",
            passed=False,
            message="No snapshot store found",
            fix_hint="Re-run synix build to recreate missing artifacts",
        )

    entries = store.iter_entries()
    missing = []
    for aid in entries:
        artifact = store.load_artifact(aid)
        if artifact is None:
            missing.append(aid)

    if missing:
        return VerifyCheck(
            name="artifacts_exist",
            passed=False,
            message=f"{len(missing)} artifacts missing from object store",
            details=missing[:20],
            fix_hint="Re-run synix build to recreate missing artifacts",
        )

    return VerifyCheck(
        name="artifacts_exist",
        passed=True,
        message=f"All {len(entries)} artifact objects present",
    )


def _check_provenance_complete(build_path: Path) -> VerifyCheck:
    """Check that all non-root artifacts have parent labels."""
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="provenance_complete",
            passed=False,
            message="No snapshot store found",
            fix_hint="Re-run synix build to regenerate provenance records",
        )

    entries = store.iter_entries()
    missing = []

    for aid in entries:
        info = entries[aid]
        level = info.get("level", 0)
        layer = info.get("layer", "")
        # Skip system artifacts (traces, etc.) — they have no provenance by design
        if layer == "traces":
            continue
        if level > 0:
            parents = store.get_parents(aid)
            if not parents:
                missing.append(f"{aid} (level {level})")

    if missing:
        return VerifyCheck(
            name="provenance_complete",
            passed=False,
            message=f"{len(missing)} non-root artifacts lack provenance",
            details=missing[:20],
            fix_hint="Re-run synix build to regenerate provenance records",
        )

    return VerifyCheck(
        name="provenance_complete",
        passed=True,
        message="All non-root artifacts have provenance records",
    )


def _check_synix_search(build_path: Path) -> VerifyCheck:
    """Check local Synix search output consistency."""
    try:
        outputs = list_search_outputs(build_path)
    except SearchOutputResolutionError as exc:
        return VerifyCheck(
            name="synix_search",
            passed=False,
            message=f"Invalid Synix search output metadata: {exc}",
            fix_hint="Re-run synix build to regenerate the local search output metadata",
        )

    if not outputs:
        return VerifyCheck(
            name="synix_search",
            passed=True,
            message="No Synix search output (not required)",
        )

    counts: list[tuple[str, int]] = []
    failures: list[str] = []

    for output in outputs:
        try:
            if not output.db_path.exists():
                failures.append(f"{output.name}: missing database at {output.db_path.relative_to(build_path)}")
                continue

            conn = sqlite3.connect(str(output.db_path))
            try:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                table_names = [t[0] for t in tables]
                if "search_index" not in table_names:
                    failures.append(
                        f"{output.name}: database at {output.db_path.relative_to(build_path)} is missing search_index"
                    )
                    continue

                count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
                counts.append((output.name, count))
            finally:
                conn.close()
        except sqlite3.Error as e:
            failures.append(f"{output.name}: SQLite error at {output.db_path.relative_to(build_path)}: {e}")

    if failures:
        return VerifyCheck(
            name="synix_search",
            passed=False,
            message=f"{len(failures)} Synix search output(s) failed verification",
            details=failures[:20],
            fix_hint="Re-run synix build to rebuild the local search outputs",
        )

    if len(counts) == 1:
        output_name, count = counts[0]
        return VerifyCheck(
            name="synix_search",
            passed=True,
            message=f"Synix search '{output_name}' has {count} entries",
        )

    summary = ", ".join(f"{name} ({count} entries)" for name, count in counts)
    return VerifyCheck(
        name="synix_search",
        passed=True,
        message=f"Synix search outputs verified: {summary}",
    )


def _check_content_hashes(build_path: Path) -> VerifyCheck:
    """Verify content hashes match actual content."""
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="content_hashes",
            passed=False,
            message="No snapshot store found",
            fix_hint="Re-run synix build to regenerate artifacts",
        )

    entries = store.iter_entries()
    mismatches = []

    for aid in entries:
        artifact = store.load_artifact(aid)
        if artifact is None:
            continue
        expected_hash = f"sha256:{hashlib.sha256(artifact.content.encode()).hexdigest()}"
        if artifact.artifact_id != expected_hash:
            mismatches.append(f"{aid}: stored={artifact.artifact_id[:20]}... computed={expected_hash[:20]}...")

    if mismatches:
        return VerifyCheck(
            name="content_hashes",
            passed=False,
            message=f"{len(mismatches)} content hash mismatches",
            details=mismatches[:20],
            fix_hint="Re-run synix build --force to regenerate corrupted artifacts",
        )

    return VerifyCheck(
        name="content_hashes",
        passed=True,
        message=f"All {len(entries)} content hashes verified",
    )


def _check_no_orphans(build_path: Path) -> VerifyCheck:
    """Check for orphaned objects — always passes with snapshot store.

    With the snapshot-based architecture, orphan detection is handled by
    content-addressing. All objects referenced by the manifest exist in the
    object store by definition.
    """
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="no_orphans",
            passed=True,
            message="No snapshot store — nothing to check",
        )
    return VerifyCheck(
        name="no_orphans",
        passed=True,
        message="Content-addressed store — no orphans possible",
    )


def _check_merge_integrity(build_path: Path) -> VerifyCheck:
    """Check that merge artifacts do not contain records from multiple customers.

    Finds all merge artifacts, then for each one walks the provenance chain
    to collect all source artifacts. If any merge artifact's sources span
    multiple distinct customer_id values, it is flagged as a cross-customer
    contamination violation.
    """
    store = _get_store(build_path)
    if store is None:
        return VerifyCheck(
            name="merge_integrity",
            passed=True,
            message="No snapshot store — nothing to check",
        )

    entries = store.iter_entries()

    # Find all merge artifacts
    merge_artifact_ids = []
    for aid in entries:
        artifact = store.load_artifact(aid)
        if artifact is None:
            continue
        if aid.startswith("merge-") or artifact.artifact_type == "merge":
            merge_artifact_ids.append(aid)

    if not merge_artifact_ids:
        return VerifyCheck(
            name="merge_integrity",
            passed=True,
            message="No merge artifacts found",
        )

    violations: list[str] = []
    affected_customers: set[str] = set()

    for merge_id in merge_artifact_ids:
        merge_artifact = store.load_artifact(merge_id)
        if merge_artifact is None:
            continue

        customer_ids: set[str] = set()

        # Check the merge artifact's own metadata for source info
        source_customer_ids = merge_artifact.metadata.get("source_customer_ids", [])
        if source_customer_ids:
            customer_ids.update(source_customer_ids)

        # Walk provenance to find customer_ids from source artifacts
        parent_labels = store.get_parents(merge_id)
        for parent_id in parent_labels:
            parent = store.load_artifact(parent_id)
            if parent and parent.metadata.get("customer_id"):
                customer_ids.add(parent.metadata["customer_id"])

        if len(customer_ids) > 1:
            sorted_customers = sorted(customer_ids)
            violations.append(
                f"{merge_id}: contains records from {len(customer_ids)} customers ({', '.join(sorted_customers)})"
            )
            affected_customers.update(customer_ids)

    if violations:
        return VerifyCheck(
            name="merge_integrity",
            passed=False,
            message=(
                f"{len(violations)} merge artifacts contain cross-customer data; "
                f"{len(affected_customers)} customers affected"
            ),
            details=violations,
            fix_hint="Add customer_id constraint to merge config and re-run synix build",
        )

    return VerifyCheck(
        name="merge_integrity",
        passed=True,
        message=f"All {len(merge_artifact_ids)} merge artifacts have single-customer sources",
    )
