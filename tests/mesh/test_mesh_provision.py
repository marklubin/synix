"""Tests for mesh provisioning — dir creation, config copy, systemd install."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from synix.mesh.provision import (
    create_mesh,
    destroy_mesh,
    list_meshes,
    provision_role,
    rotate_token,
)


class TestCreateMesh:
    def test_creates_mesh_dir(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test pipeline")
        config = create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        assert (tmp_path / "test-mesh").is_dir()
        assert config.name == "test-mesh"

    def test_creates_config_file(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test pipeline")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        config_path = tmp_path / "test-mesh" / "synix-mesh.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'name = "test-mesh"' in content
        assert "msh_" in content

    def test_copies_pipeline(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# my pipeline code")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        copied = tmp_path / "test-mesh" / "pipeline.py"
        assert copied.exists()
        assert copied.read_text() == "# my pipeline code"

    def test_creates_state_json(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        state_path = tmp_path / "test-mesh" / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["term"]["counter"] == 0

    def test_generates_unique_token(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        c1 = create_mesh("mesh-1", pipeline, mesh_root=tmp_path)
        c2 = create_mesh("mesh-2", pipeline, mesh_root=tmp_path)
        assert c1.token != c2.token
        assert c1.token.startswith("msh_")

    def test_duplicate_name_raises(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            create_mesh("test-mesh", pipeline, mesh_root=tmp_path)


class TestProvisionRole:
    def _setup_mesh(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        return create_mesh("test-mesh", pipeline, mesh_root=tmp_path)

    @patch("synix.mesh.provision._install_systemd_unit")
    def test_server_creates_dirs(self, mock_systemd, tmp_path):
        self._setup_mesh(tmp_path)
        provision_role("test-mesh", "server", mesh_root=tmp_path)
        mesh_dir = tmp_path / "test-mesh"
        assert (mesh_dir / "server" / "sessions").is_dir()
        assert (mesh_dir / "server" / "build").is_dir()
        assert (mesh_dir / "server" / "bundles").is_dir()

    @patch("synix.mesh.provision._install_systemd_unit")
    def test_client_creates_dirs(self, mock_systemd, tmp_path):
        self._setup_mesh(tmp_path)
        provision_role("test-mesh", "client", server_url="http://server:7433", mesh_root=tmp_path)
        mesh_dir = tmp_path / "test-mesh"
        assert (mesh_dir / "client" / "build").is_dir()

    @patch("synix.mesh.provision._install_systemd_unit")
    def test_updates_state_json(self, mock_systemd, tmp_path):
        self._setup_mesh(tmp_path)
        provision_role("test-mesh", "server", mesh_root=tmp_path)
        state = json.loads((tmp_path / "test-mesh" / "state.json").read_text())
        assert state["role"] == "server"
        assert state["my_hostname"]  # should be set to socket.gethostname()

    @patch("synix.mesh.provision._install_systemd_unit")
    def test_client_stores_server_url(self, mock_systemd, tmp_path):
        self._setup_mesh(tmp_path)
        provision_role("test-mesh", "client", server_url="http://obispo:7433", mesh_root=tmp_path)
        state = json.loads((tmp_path / "test-mesh" / "state.json").read_text())
        assert state["server_url"] == "http://obispo:7433"

    @patch("synix.mesh.provision._install_systemd_unit")
    def test_idempotent_provision(self, mock_systemd, tmp_path):
        self._setup_mesh(tmp_path)
        provision_role("test-mesh", "server", mesh_root=tmp_path)
        provision_role("test-mesh", "server", mesh_root=tmp_path)  # should not raise

    def test_missing_mesh_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            provision_role("nonexistent", "server", mesh_root=tmp_path)


class TestDestroyMesh:
    def test_removes_mesh_dir(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        assert (tmp_path / "test-mesh").exists()
        destroy_mesh("test-mesh", mesh_root=tmp_path)
        assert not (tmp_path / "test-mesh").exists()

    def test_destroy_nonexistent_no_error(self, tmp_path):
        destroy_mesh("nonexistent", mesh_root=tmp_path)  # should not raise


class TestListMeshes:
    def test_empty(self, tmp_path):
        assert list_meshes(mesh_root=tmp_path) == []

    def test_lists_created_meshes(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        create_mesh("mesh-a", pipeline, mesh_root=tmp_path)
        create_mesh("mesh-b", pipeline, mesh_root=tmp_path)
        meshes = list_meshes(mesh_root=tmp_path)
        names = [m["name"] for m in meshes]
        assert "mesh-a" in names
        assert "mesh-b" in names

    def test_includes_role(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        meshes = list_meshes(mesh_root=tmp_path)
        assert meshes[0]["role"] == ""  # not provisioned yet


class TestRotateToken:
    def test_rotates_token(self, tmp_path):
        pipeline = tmp_path / "pipeline.py"
        pipeline.write_text("# test")
        config = create_mesh("test-mesh", pipeline, mesh_root=tmp_path)
        old_token = config.token
        new_token = rotate_token("test-mesh", mesh_root=tmp_path)
        assert new_token != old_token
        assert new_token.startswith("msh_")
        # Verify written to file
        content = (tmp_path / "test-mesh" / "synix-mesh.toml").read_text()
        assert new_token in content

    def test_missing_mesh_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            rotate_token("nonexistent", mesh_root=tmp_path)
