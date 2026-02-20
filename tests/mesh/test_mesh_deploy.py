"""Tests for mesh deploy hook runner."""

from __future__ import annotations

import subprocess

import pytest

from synix.mesh.deploy import run_deploy_hooks


class TestRunDeployHooks:
    def test_successful_command(self, tmp_path):
        """A simple echo command runs without error."""
        run_deploy_hooks(["echo hello"], tmp_path)

    def test_build_dir_substitution(self, tmp_path):
        """The {build_dir} placeholder is replaced with the actual path."""
        marker = tmp_path / "proof.txt"
        run_deploy_hooks(
            ["echo {build_dir} > {build_dir}/proof.txt"],
            tmp_path,
        )
        assert marker.exists()
        assert str(tmp_path) in marker.read_text()

    def test_nonzero_exit_raises_runtime_error(self, tmp_path):
        """Non-zero exit code raises RuntimeError."""
        with pytest.raises(RuntimeError, match="rc=1"):
            run_deploy_hooks(["exit 1"], tmp_path)

    def test_timeout_raises(self, tmp_path, monkeypatch):
        """A command exceeding the timeout raises subprocess.TimeoutExpired."""
        monkeypatch.setattr("synix.mesh.deploy.DEPLOY_TIMEOUT", 1)
        with pytest.raises(subprocess.TimeoutExpired):
            run_deploy_hooks(["sleep 10"], tmp_path)

    def test_multiple_commands_sequential(self, tmp_path):
        """Multiple commands run in order — second sees first's output."""
        run_deploy_hooks(
            [
                "echo step1 > steps.txt",
                "echo step2 >> steps.txt",
            ],
            tmp_path,
        )
        content = (tmp_path / "steps.txt").read_text()
        assert "step1" in content
        assert "step2" in content

    def test_cwd_is_build_dir(self, tmp_path):
        """Commands execute with cwd set to build_dir."""
        run_deploy_hooks(["pwd > cwd.txt"], tmp_path)
        cwd_output = (tmp_path / "cwd.txt").read_text().strip()
        assert cwd_output == str(tmp_path)

    def test_empty_command_list(self, tmp_path):
        """Empty command list is a no-op."""
        run_deploy_hooks([], tmp_path)

    def test_failure_stops_at_first_bad_command(self, tmp_path):
        """If a command fails, subsequent commands do not run."""
        with pytest.raises(RuntimeError):
            run_deploy_hooks(
                [
                    "echo first > order.txt",
                    "exit 1",
                    "echo third >> order.txt",
                ],
                tmp_path,
            )
        content = (tmp_path / "order.txt").read_text()
        assert "first" in content
        assert "third" not in content
