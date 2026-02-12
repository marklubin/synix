"""Tests for template sync — verifies bundled templates match examples."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
TEMPLATES_DIR = REPO_ROOT / "src" / "synix" / "templates"

# Files that sync-templates copies from examples into templates
# (.env.example is shared at templates root, not per-template)
USER_FACING_FILES = ["pipeline.py", "README.md"]


def _template_dirs():
    """Return list of template directories that exist."""
    if not TEMPLATES_DIR.is_dir():
        return []
    return sorted(d for d in TEMPLATES_DIR.iterdir() if d.is_dir())


@pytest.mark.skipif(
    not TEMPLATES_DIR.is_dir() or not any(TEMPLATES_DIR.iterdir()),
    reason="Templates not synced yet — run scripts/sync-templates first",
)
class TestTemplatesMatchExamples:
    """Verify src/synix/templates/ is in sync with examples/."""

    def test_every_template_has_matching_example(self):
        """Each template dir should correspond to an example."""
        for template_dir in _template_dirs():
            example_dir = EXAMPLES_DIR / template_dir.name
            assert example_dir.is_dir(), (
                f"Template {template_dir.name} has no matching example at {example_dir}. "
                f"Run scripts/sync-templates to regenerate."
            )

    def test_user_facing_files_match(self):
        """User-facing files in templates should match their example source."""
        for template_dir in _template_dirs():
            example_dir = EXAMPLES_DIR / template_dir.name
            if not example_dir.is_dir():
                continue

            for filename in USER_FACING_FILES:
                template_file = template_dir / filename
                example_file = example_dir / filename
                if not example_file.exists():
                    continue
                assert template_file.exists(), (
                    f"Template {template_dir.name} is missing {filename}. Run scripts/sync-templates."
                )
                assert template_file.read_text() == example_file.read_text(), (
                    f"Template {template_dir.name}/{filename} differs from example. Run scripts/sync-templates."
                )

            # Also check pipeline_*.py files (example 01 has two)
            for pipeline_file in example_dir.glob("pipeline_*.py"):
                template_file = template_dir / pipeline_file.name
                assert template_file.exists(), (
                    f"Template {template_dir.name} is missing {pipeline_file.name}. Run scripts/sync-templates."
                )
                assert template_file.read_text() == pipeline_file.read_text(), (
                    f"Template {template_dir.name}/{pipeline_file.name} differs from example. "
                    f"Run scripts/sync-templates."
                )

    def test_sources_dir_matches(self):
        """sources/ directory contents should match between template and example."""
        for template_dir in _template_dirs():
            example_dir = EXAMPLES_DIR / template_dir.name
            if not example_dir.is_dir():
                continue

            example_sources = example_dir / "sources"
            template_sources = template_dir / "sources"

            if not example_sources.is_dir():
                continue

            assert template_sources.is_dir(), (
                f"Template {template_dir.name} is missing sources/. Run scripts/sync-templates."
            )

            # Compare all files recursively
            for example_file in sorted(example_sources.rglob("*")):
                if example_file.is_dir():
                    continue
                relative = example_file.relative_to(example_sources)
                template_file = template_sources / relative
                assert template_file.exists(), (
                    f"Template {template_dir.name}/sources/{relative} is missing. Run scripts/sync-templates."
                )
                assert template_file.read_text() == example_file.read_text(), (
                    f"Template {template_dir.name}/sources/{relative} differs. Run scripts/sync-templates."
                )

    def test_shared_env_example_exists(self):
        """Shared .env.example should exist at templates root."""
        shared_env = TEMPLATES_DIR / ".env.example"
        assert shared_env.exists(), (
            "Missing src/synix/templates/.env.example — this is the shared .env.example for all templates."
        )
        content = shared_env.read_text()
        assert "ANTHROPIC_API_KEY" in content

    def test_no_per_template_env_example(self):
        """Per-template .env.example should not exist (use shared one)."""
        for template_dir in _template_dirs():
            assert not (template_dir / ".env.example").exists(), (
                f"Template {template_dir.name} has per-template .env.example — "
                f"use shared templates/.env.example instead."
            )

    def test_no_non_user_facing_files_in_templates(self):
        """Templates should NOT contain case.py, cassettes/, golden/, etc."""
        for template_dir in _template_dirs():
            assert not (template_dir / "case.py").exists(), (
                f"Template {template_dir.name} contains case.py — this is not a user-facing file."
            )
            assert not (template_dir / "cassettes").exists(), (
                f"Template {template_dir.name} contains cassettes/ — this is not a user-facing file."
            )
            assert not (template_dir / "golden").exists(), (
                f"Template {template_dir.name} contains golden/ — this is not a user-facing file."
            )
