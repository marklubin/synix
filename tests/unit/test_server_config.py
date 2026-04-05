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
    """Auto-build section is parsed correctly (backward-compatible)."""
    cfg = load_config(str(toml_file))

    assert cfg.auto_build.enabled is True
    assert cfg.auto_build.window == 30  # default, old keys ignored


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
    assert cfg.auto_build.window == 30
    assert cfg.vllm.enabled is False
    assert cfg.vllm.model == "Qwen/Qwen2.5-3B-Instruct"
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


def test_vllm_config(tmp_path: Path):
    """vLLM section is parsed correctly."""
    content = textwrap.dedent("""\
        [server]
        project_dir = "/tmp/test"

        [vllm]
        enabled = true
        model = "Qwen/Qwen2.5-3B-Instruct"
        gpu_device = 0
        port = 8100
        max_model_len = 2048
        gpu_memory_utilization = 0.85
        startup_timeout = 90
    """)
    p = tmp_path / "cfg.toml"
    p.write_text(content)

    cfg = load_config(str(p))
    assert cfg.vllm.enabled is True
    assert cfg.vllm.model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.vllm.gpu_device == 0
    assert cfg.vllm.port == 8100
    assert cfg.vllm.max_model_len == 2048
    assert cfg.vllm.gpu_memory_utilization == 0.85
    assert cfg.vllm.startup_timeout == 90


def test_build_queue_window(tmp_path: Path):
    """Build queue window setting is parsed correctly."""
    content = textwrap.dedent("""\
        [server]
        project_dir = "/tmp/test"

        [auto_build]
        enabled = true
        window = 60
    """)
    p = tmp_path / "cfg.toml"
    p.write_text(content)

    cfg = load_config(str(p))
    assert cfg.auto_build.enabled is True
    assert cfg.auto_build.window == 60
