"""Pipeline verification — check integrity of a completed build."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker


@dataclass
class VerifyCheck:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


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
                }
                for c in self.checks
            ],
        }


def verify_build(build_dir: str | Path, checks: list[str] | None = None) -> VerifyResult:
    """Run verification checks on a completed build.

    Args:
        build_dir: Path to the build directory.
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
        "search_index": _check_search_index,
        "content_hashes": _check_content_hashes,
        "no_orphans": _check_no_orphans,
        "merge_integrity": _check_merge_integrity,
    }

    checks_to_run = checks if checks else list(all_checks.keys())

    for check_name in checks_to_run:
        if check_name in all_checks:
            check_result = all_checks[check_name](build_path)
            result.checks.append(check_result)
        else:
            result.checks.append(VerifyCheck(
                name=check_name,
                passed=False,
                message=f"Unknown check: {check_name}",
            ))

    return result


def _check_build_exists(build_path: Path) -> VerifyCheck:
    """Check that the build directory exists and has a manifest."""
    if not build_path.exists():
        return VerifyCheck(
            name="build_exists",
            passed=False,
            message="Build directory does not exist",
        )
    manifest_path = build_path / "manifest.json"
    if not manifest_path.exists():
        return VerifyCheck(
            name="build_exists",
            passed=False,
            message="No manifest.json found in build directory",
        )
    return VerifyCheck(
        name="build_exists",
        passed=True,
        message="Build directory exists with manifest",
    )


def _check_manifest_valid(build_path: Path) -> VerifyCheck:
    """Check that manifest.json is valid JSON with expected structure."""
    manifest_path = build_path / "manifest.json"
    if not manifest_path.exists():
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message="No manifest.json found",
        )
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message=f"Invalid JSON in manifest: {e}",
        )

    if not isinstance(data, dict):
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message="Manifest is not a JSON object",
        )

    issues = []
    for aid, entry in data.items():
        if not isinstance(entry, dict):
            issues.append(f"{aid}: entry is not a dict")
            continue
        for key in ("path", "content_hash", "layer", "level"):
            if key not in entry:
                issues.append(f"{aid}: missing key '{key}'")

    if issues:
        return VerifyCheck(
            name="manifest_valid",
            passed=False,
            message=f"Manifest has {len(issues)} structural issues",
            details=issues[:20],
        )

    return VerifyCheck(
        name="manifest_valid",
        passed=True,
        message=f"Manifest valid with {len(data)} artifacts",
    )


def _check_artifacts_exist(build_path: Path) -> VerifyCheck:
    """Check that all manifest entries have corresponding files on disk."""
    manifest_path = build_path / "manifest.json"
    if not manifest_path.exists():
        return VerifyCheck(
            name="artifacts_exist",
            passed=False,
            message="No manifest.json found",
        )

    manifest = json.loads(manifest_path.read_text())
    missing = []
    for aid, entry in manifest.items():
        artifact_path = build_path / entry.get("path", "")
        if not artifact_path.exists():
            missing.append(f"{aid}: {entry.get('path', '?')}")

    if missing:
        return VerifyCheck(
            name="artifacts_exist",
            passed=False,
            message=f"{len(missing)} artifacts missing from disk",
            details=missing[:20],
        )

    return VerifyCheck(
        name="artifacts_exist",
        passed=True,
        message=f"All {len(manifest)} artifact files present",
    )


def _check_provenance_complete(build_path: Path) -> VerifyCheck:
    """Check that all non-root artifacts have provenance records."""
    manifest_path = build_path / "manifest.json"
    provenance_path = build_path / "provenance.json"

    if not manifest_path.exists():
        return VerifyCheck(
            name="provenance_complete",
            passed=False,
            message="No manifest.json found",
        )

    manifest = json.loads(manifest_path.read_text())

    if not provenance_path.exists():
        # Check if there are any non-root artifacts
        non_root = [aid for aid, e in manifest.items() if e.get("level", 0) > 0]
        if non_root:
            return VerifyCheck(
                name="provenance_complete",
                passed=False,
                message=f"No provenance.json but {len(non_root)} non-root artifacts exist",
            )
        return VerifyCheck(
            name="provenance_complete",
            passed=True,
            message="No non-root artifacts, provenance not required",
        )

    provenance = json.loads(provenance_path.read_text())
    missing = []
    for aid, entry in manifest.items():
        level = entry.get("level", 0)
        if level > 0 and aid not in provenance:
            missing.append(f"{aid} (level {level})")

    if missing:
        return VerifyCheck(
            name="provenance_complete",
            passed=False,
            message=f"{len(missing)} non-root artifacts lack provenance",
            details=missing[:20],
        )

    return VerifyCheck(
        name="provenance_complete",
        passed=True,
        message="All non-root artifacts have provenance records",
    )


def _check_search_index(build_path: Path) -> VerifyCheck:
    """Check search index consistency."""
    db_path = build_path / "search.db"
    if not db_path.exists():
        return VerifyCheck(
            name="search_index",
            passed=True,
            message="No search index (not required)",
        )

    try:
        conn = sqlite3.connect(str(db_path))
        # Check table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        if "search_index" not in table_names:
            conn.close()
            return VerifyCheck(
                name="search_index",
                passed=False,
                message="search.db exists but search_index table missing",
            )

        # Count rows
        count = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        conn.close()

        return VerifyCheck(
            name="search_index",
            passed=True,
            message=f"Search index has {count} entries",
        )
    except sqlite3.Error as e:
        return VerifyCheck(
            name="search_index",
            passed=False,
            message=f"Search index error: {e}",
        )


def _check_content_hashes(build_path: Path) -> VerifyCheck:
    """Verify content hashes match actual content."""
    import hashlib

    store = ArtifactStore(build_path)
    manifest = store._manifest
    mismatches = []

    for aid in manifest:
        artifact = store.load_artifact(aid)
        if artifact is None:
            continue
        expected_hash = f"sha256:{hashlib.sha256(artifact.content.encode()).hexdigest()}"
        if artifact.content_hash != expected_hash:
            mismatches.append(
                f"{aid}: stored={artifact.content_hash[:20]}... computed={expected_hash[:20]}..."
            )

    if mismatches:
        return VerifyCheck(
            name="content_hashes",
            passed=False,
            message=f"{len(mismatches)} content hash mismatches",
            details=mismatches[:20],
        )

    return VerifyCheck(
        name="content_hashes",
        passed=True,
        message=f"All {len(manifest)} content hashes verified",
    )


def _check_no_orphans(build_path: Path) -> VerifyCheck:
    """Check for orphaned artifact files not in manifest."""
    manifest_path = build_path / "manifest.json"
    if not manifest_path.exists():
        return VerifyCheck(
            name="no_orphans",
            passed=True,
            message="No manifest — nothing to check",
        )

    manifest = json.loads(manifest_path.read_text())
    manifest_paths = {entry["path"] for entry in manifest.values()}

    # Find all artifact JSON files in layer directories
    orphans = []
    for layer_dir in build_path.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer"):
            continue
        for json_file in layer_dir.glob("*.json"):
            rel_path = f"{layer_dir.name}/{json_file.name}"
            if rel_path not in manifest_paths:
                orphans.append(rel_path)

    if orphans:
        return VerifyCheck(
            name="no_orphans",
            passed=False,
            message=f"{len(orphans)} orphaned artifact files",
            details=orphans[:20],
        )

    return VerifyCheck(
        name="no_orphans",
        passed=True,
        message="No orphaned artifact files",
    )


def _check_merge_integrity(build_path: Path) -> VerifyCheck:
    """Check that merge artifacts do not contain records from multiple customers.

    Finds all merge artifacts in the manifest, then for each one walks the
    provenance chain to collect all source artifacts. If any merge artifact's
    sources span multiple distinct customer_id values, it is flagged as a
    cross-customer contamination violation.
    """
    manifest_path = build_path / "manifest.json"
    provenance_path = build_path / "provenance.json"

    if not manifest_path.exists():
        return VerifyCheck(
            name="merge_integrity",
            passed=True,
            message="No manifest — nothing to check",
        )

    manifest = json.loads(manifest_path.read_text())

    # Find all merge artifacts
    merge_artifact_ids = [
        aid for aid, entry in manifest.items()
        if aid.startswith("merge-") or entry.get("artifact_type") == "merge"
    ]

    if not merge_artifact_ids:
        return VerifyCheck(
            name="merge_integrity",
            passed=True,
            message="No merge artifacts found",
        )

    # Load provenance if available
    provenance: dict = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())

    store = ArtifactStore(build_path)

    violations: list[str] = []
    affected_customers: set[str] = set()

    for merge_id in merge_artifact_ids:
        # Get the merge artifact to check its metadata
        merge_artifact = store.load_artifact(merge_id)
        if merge_artifact is None:
            continue

        # Collect customer_ids from source artifacts via provenance
        customer_ids: set[str] = set()

        # First, check the merge artifact's own metadata for source info
        source_customer_ids = merge_artifact.metadata.get("source_customer_ids", [])
        if source_customer_ids:
            customer_ids.update(source_customer_ids)

        # Also walk provenance to find customer_ids from source artifacts
        if merge_id in provenance:
            parent_ids = provenance[merge_id].get("parent_artifact_ids", [])
            for parent_id in parent_ids:
                parent = store.load_artifact(parent_id)
                if parent and parent.metadata.get("customer_id"):
                    customer_ids.add(parent.metadata["customer_id"])

        # Flag if multiple customers
        if len(customer_ids) > 1:
            sorted_customers = sorted(customer_ids)
            violations.append(
                f"{merge_id}: contains records from {len(customer_ids)} customers "
                f"({', '.join(sorted_customers)})"
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
        )

    return VerifyCheck(
        name="merge_integrity",
        passed=True,
        message=f"All {len(merge_artifact_ids)} merge artifacts have single-customer sources",
    )
