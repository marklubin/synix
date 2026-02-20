"""Tests for mesh config loading and mesh dir management."""

from __future__ import annotations

from pathlib import Path

import pytest

from synix.mesh.config import (
    ClusterConfig,
    MeshConfig,
    ServerConfig,
    ensure_mesh_dirs,
    load_mesh_config,
)


class TestLoadMeshConfig:
    def test_load_full_config(self, mesh_toml_file):
        config = load_mesh_config(mesh_toml_file)
        assert config.name == "test-mesh"
        assert config.token == "msh_testtoken123"
        assert config.pipeline_path == "./pipeline.py"
        assert config.server.port == 7433
        assert config.server.build_min_interval == 300
        assert config.client.scan_interval == 30
        assert config.source.watch_dir == "/tmp/watch"
        assert config.source.patterns == ["**/*.jsonl"]
        assert config.cluster.leader_candidates == ["obispo", "salinas", "oxnard"]
        assert config.bundle.include == ["manifest.json", "context.md"]
        assert config.deploy.server_commands == ["echo server-deploy"]
        assert config.deploy.client_commands == ["echo client-deploy"]
        assert config.notifications.webhook_url == "http://localhost:8080/notifications/push"

    def test_missing_name_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text("[mesh]\n")
        with pytest.raises(ValueError, match="Missing required field.*name"):
            load_mesh_config(toml_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Config file not found"):
            load_mesh_config(tmp_path / "nonexistent.toml")

    def test_defaults_applied(self, tmp_path):
        toml_path = tmp_path / "minimal.toml"
        toml_path.write_text('[mesh]\nname = "minimal"\n')
        config = load_mesh_config(toml_path)
        assert config.name == "minimal"
        assert config.token == ""
        assert config.server.port == 7433
        assert config.client.scan_interval == 30
        assert config.source.env_var == "SYNIX_SOURCE_DIR"

    def test_env_var_overrides_port(self, mesh_toml_file, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_PORT", "9999")
        config = load_mesh_config(mesh_toml_file)
        assert config.server.port == 9999

    def test_env_var_overrides_watch_dir(self, mesh_toml_file, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_SOURCE_WATCH_DIR", "/custom/watch")
        config = load_mesh_config(mesh_toml_file)
        assert config.source.watch_dir == "/custom/watch"


class TestLegacyConfigKeys:
    def test_build_batch_threshold_uses_default(self, tmp_path):
        """Legacy build_batch_threshold is ignored; default quiet_period is used."""
        toml_path = tmp_path / "legacy.toml"
        toml_path.write_text('[mesh]\nname = "legacy"\n\n[server]\nbuild_batch_threshold = 45\n')
        config = load_mesh_config(toml_path)
        # Old value (45) is NOT used — semantics are incompatible (count vs seconds)
        assert config.server.build_quiet_period == ServerConfig.build_quiet_period

    def test_quiet_period_takes_precedence_over_legacy(self, tmp_path):
        """If both keys are present, build_quiet_period wins."""
        toml_path = tmp_path / "both.toml"
        toml_path.write_text('[mesh]\nname = "both"\n\n[server]\nbuild_quiet_period = 90\nbuild_batch_threshold = 45\n')
        config = load_mesh_config(toml_path)
        assert config.server.build_quiet_period == 90

    def test_legacy_key_logs_warning(self, tmp_path, caplog):
        """Legacy key emits a removal warning."""
        toml_path = tmp_path / "legacy-warn.toml"
        toml_path.write_text('[mesh]\nname = "warn"\n\n[server]\nbuild_batch_threshold = 30\n')
        import logging

        with caplog.at_level(logging.WARNING, logger="synix.mesh.config"):
            load_mesh_config(toml_path)
        assert "build_batch_threshold" in caplog.text
        assert "removed" in caplog.text.lower()


class TestMeshDir:
    def test_mesh_dir_default(self):
        config = MeshConfig(name="test")
        assert config.mesh_dir == Path.home() / ".synix-mesh" / "test"

    def test_mesh_dir_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="custom")
        assert config.mesh_dir == tmp_path / "custom"


class TestEnsureMeshDirs:
    def test_server_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test-dirs")
        ensure_mesh_dirs(config, "server")
        mesh_dir = tmp_path / "test-dirs"
        assert (mesh_dir / "server" / "sessions").is_dir()
        assert (mesh_dir / "server" / "build").is_dir()
        assert (mesh_dir / "server" / "bundles").is_dir()

    def test_client_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test-dirs")
        ensure_mesh_dirs(config, "client")
        mesh_dir = tmp_path / "test-dirs"
        assert (mesh_dir / "client" / "build").is_dir()

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test-dirs")
        ensure_mesh_dirs(config, "server")
        ensure_mesh_dirs(config, "server")  # should not raise
        assert (tmp_path / "test-dirs" / "server" / "sessions").is_dir()


class TestClusterConfigHash:
    def test_hash_deterministic(self):
        c1 = ClusterConfig(leader_candidates=["a", "b", "c"])
        c2 = ClusterConfig(leader_candidates=["a", "b", "c"])
        assert c1.config_hash == c2.config_hash

    def test_hash_ordering_matters(self):
        c1 = ClusterConfig(leader_candidates=["a", "b", "c"])
        c2 = ClusterConfig(leader_candidates=["c", "b", "a"])
        assert c1.config_hash != c2.config_hash

    def test_hash_empty_candidates(self):
        c = ClusterConfig(leader_candidates=[])
        assert c.config_hash  # should return a valid hash
