"""Shared fixtures for mesh tests."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.mesh


@pytest.fixture
def mesh_token():
    """A pre-generated mesh token."""
    return "msh_" + "a1b2c3d4e5f6" * 4 + "deadbeef"


@pytest.fixture
def mesh_config_dict():
    """Raw dict representing a synix-mesh.toml."""
    return {
        "mesh": {"name": "test-mesh", "token": "msh_testtoken123"},
        "pipeline": {"path": "./pipeline.py"},
        "source": {
            "watch_dir": "/tmp/watch",
            "patterns": ["**/*.jsonl"],
            "exclude": ["**/subagents/**"],
            "env_var": "SYNIX_SOURCE_DIR",
        },
        "server": {
            "port": 7433,
            "build_min_interval": 300,
            "build_batch_threshold": 5,
            "build_max_delay": 900,
        },
        "client": {
            "scan_interval": 30,
            "pull_interval": 60,
            "heartbeat_interval": 120,
        },
        "cluster": {
            "leader_candidates": ["obispo", "salinas", "oxnard"],
            "leader_timeout": 360,
        },
        "bundle": {
            "include": ["manifest.json", "context.md"],
            "exclude": ["search.db"],
        },
        "deploy": {
            "server": {"commands": ["echo server-deploy"]},
            "client": {"commands": ["echo client-deploy"]},
        },
        "notifications": {
            "webhook_url": "http://localhost:8080/notifications/push",
            "source": "test-mesh",
        },
    }


def _write_toml(data: dict, path: Path) -> Path:
    """Write a dict as TOML to a file. Simple serializer for test data."""
    import tomllib  # noqa: F401 — just to confirm stdlib availability

    lines = []

    def _write_section(prefix: str, d: dict):
        scalars = {}
        tables = {}
        for k, v in d.items():
            if isinstance(v, dict):
                tables[k] = v
            else:
                scalars[k] = v
        if scalars:
            if prefix:
                lines.append(f"[{prefix}]")
            for k, v in scalars.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, int):
                    lines.append(f"{k} = {v}")
                elif isinstance(v, list):
                    items = ", ".join(json.dumps(i) for i in v)
                    lines.append(f"{k} = [{items}]")
            lines.append("")
        for k, v in tables.items():
            sub = f"{prefix}.{k}" if prefix else k
            _write_section(sub, v)

    _write_section("", data)
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def mesh_toml_file(tmp_path, mesh_config_dict):
    """Write mesh config dict to a temp TOML file."""
    return _write_toml(mesh_config_dict, tmp_path / "synix-mesh.toml")


@pytest.fixture
def mesh_data_dir(tmp_path):
    """Create a temp mesh data directory structure."""
    mesh_dir = tmp_path / "mesh-data"
    mesh_dir.mkdir()
    (mesh_dir / "server" / "sessions").mkdir(parents=True)
    (mesh_dir / "server" / "build").mkdir(parents=True)
    (mesh_dir / "server" / "bundles").mkdir(parents=True)
    (mesh_dir / "client" / "build").mkdir(parents=True)
    return mesh_dir


@pytest.fixture
def sample_session_file(tmp_path):
    """A sample JSONL session file."""
    content = textwrap.dedent("""\
        {"role": "user", "content": "Hello, how are you?"}
        {"role": "assistant", "content": "I'm doing well, thanks for asking!"}
        {"role": "user", "content": "Tell me about Python."}
        {"role": "assistant", "content": "Python is a versatile programming language..."}
    """)
    path = tmp_path / "session-001.jsonl"
    path.write_text(content)
    return path


@pytest.fixture
def auth_headers_fixture(mesh_token):
    """Pre-built auth headers dict."""
    return {"Authorization": f"Bearer {mesh_token}", "X-Mesh-Node": "test-node"}


@pytest.fixture
def mock_build_artifacts(tmp_path):
    """Create mock build artifacts in a temp dir."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "manifest.json").write_text('{"version": 1, "artifacts": []}')
    (build_dir / "context.md").write_text("# Test Context\n\nThis is test content.")
    layer_dir = build_dir / "layer2-episodes"
    layer_dir.mkdir()
    (layer_dir / "ep-001.json").write_text('{"label": "ep-001", "content": "test episode"}')
    return build_dir
