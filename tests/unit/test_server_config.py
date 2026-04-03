"""Tests for synix.server.config — TOML loading and defaults."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from synix.server.config import ServerConfig, load_config


@pytest.fixture()
def toml_file(tmp_path: Path) -> Path:
    """Write a full config TOML and return its path."""
    content = textwrap.dedent("""\
        [server]
        project_dir = "/srv/synix/project"
        pipeline_path = "pipeline.py"
        mcp_port = 8200
        viewer_port = 9471

        [buckets.sessions]
        dir = "sources/sessions"
        patterns = ["**/*.jsonl.gz", "**/*.jsonl"]
        description = "Claude Code session transcripts"

        [buckets.documents]
        dir = "sources/documents"
        patterns = ["**/*.md", "**/*.txt", "**/*.pdf"]
        description = "Notes, specs, memos"

        [buckets.reports]
        dir = "sources/reports"
        patterns = ["**/*.md"]
        description = "Cron and automated reports"

        [auto_build]
        enabled = true
        scan_interval = 60
        cooldown = 300
    """)
    p = tmp_path / "synix-server.toml"
    p.write_text(content)
    return p


def test_load_full_config(toml_file: Path):
    """Loading a fully specified TOML produces correct values."""
    cfg = load_config(str(toml_file))

    assert isinstance(cfg, ServerConfig)
    assert cfg.project_dir == "/srv/synix/project"
    assert cfg.pipeline_path == "pipeline.py"
    assert cfg.mcp_port == 8200
    assert cfg.viewer_port == 9471


def test_bucket_parsing(toml_file: Path):
    """All buckets are parsed with correct names, dirs, patterns, descriptions."""
    cfg = load_config(str(toml_file))

    assert len(cfg.buckets) == 3

    by_name = {b.name: b for b in cfg.buckets}
    assert "sessions" in by_name
    assert "documents" in by_name
    assert "reports" in by_name

    sessions = by_name["sessions"]
    assert sessions.dir == "sources/sessions"
    assert sessions.patterns == ["**/*.jsonl.gz", "**/*.jsonl"]
    assert sessions.description == "Claude Code session transcripts"


def test_auto_build_config(toml_file: Path):
    """Auto-build section is parsed correctly."""
    cfg = load_config(str(toml_file))

    assert cfg.auto_build.enabled is True
    assert cfg.auto_build.scan_interval == 60
    assert cfg.auto_build.cooldown == 300


def test_defaults_when_optional_missing(tmp_path: Path):
    """Optional fields use defaults when not specified."""
    content = textwrap.dedent("""\
        [server]
        project_dir = "/tmp/test-project"
    """)
    p = tmp_path / "minimal.toml"
    p.write_text(content)

    cfg = load_config(str(p))

    assert cfg.project_dir == "/tmp/test-project"
    assert cfg.pipeline_path == "pipeline.py"
    assert cfg.mcp_port == 8200
    assert cfg.viewer_port == 9471
    assert cfg.viewer_host == "0.0.0.0"
    assert cfg.buckets == []
    assert cfg.auto_build.enabled is True
    assert cfg.auto_build.scan_interval == 60
    assert cfg.auto_build.cooldown == 300
    assert cfg.allowed_hosts == []


def test_missing_project_dir(tmp_path: Path):
    """Missing project_dir raises KeyError."""
    content = textwrap.dedent("""\
        [server]
        mcp_port = 9999
    """)
    p = tmp_path / "bad.toml"
    p.write_text(content)

    with pytest.raises(KeyError, match="project_dir"):
        load_config(str(p))


def test_missing_file():
    """Non-existent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/synix-server.toml")


def test_bucket_default_patterns(tmp_path: Path):
    """Bucket without explicit patterns gets default ['**/*']."""
    content = textwrap.dedent("""\
        [server]
        project_dir = "/tmp/test"

        [buckets.misc]
        dir = "sources/misc"
    """)
    p = tmp_path / "cfg.toml"
    p.write_text(content)

    cfg = load_config(str(p))
    assert len(cfg.buckets) == 1
    assert cfg.buckets[0].name == "misc"
    assert cfg.buckets[0].patterns == ["**/*"]
    assert cfg.buckets[0].description == ""
