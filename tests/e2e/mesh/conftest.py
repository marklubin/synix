"""E2E mesh test fixtures — creates a real Starlette test server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from synix.mesh.auth import auth_headers
from synix.mesh.config import load_mesh_config
from synix.mesh.server import create_app

MESH_TOKEN = "msh_testtoken1234567890abcdef1234567890abcdef1234567890abcdef12345678"
TOY_PIPELINE_DIR = Path(__file__).parent / "toy-pipeline"


@pytest.fixture
def mesh_config(tmp_path, monkeypatch):
    """Create a mesh config from the toy pipeline for testing."""
    monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))

    # Set up mesh directory structure
    mesh_dir = tmp_path / "test-mesh"
    mesh_dir.mkdir()
    server_dir = mesh_dir / "server"
    (server_dir / "sessions").mkdir(parents=True)
    (server_dir / "build").mkdir(parents=True)
    (server_dir / "bundles").mkdir(parents=True)

    # Copy config
    import shutil

    shutil.copy(TOY_PIPELINE_DIR / "synix-mesh.toml", mesh_dir / "synix-mesh.toml")
    shutil.copy(TOY_PIPELINE_DIR / "pipeline.py", mesh_dir / "pipeline.py")

    # Write state
    state = {
        "term": {"counter": 1, "leader_id": "test-server"},
        "candidates": [],
        "config_hash": "",
        "role": "server",
        "server_url": "http://localhost:7433",
        "my_hostname": "test-server",
    }
    (mesh_dir / "state.json").write_text(json.dumps(state))

    return load_mesh_config(mesh_dir / "synix-mesh.toml")


@pytest.fixture
def test_client(mesh_config):
    """Starlette test client with auth headers."""
    app = create_app(mesh_config)
    return TestClient(app)


@pytest.fixture
def authed_headers():
    """Pre-built auth headers for the test token."""
    return auth_headers(MESH_TOKEN, node_name="test-node")
