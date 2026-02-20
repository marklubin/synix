"""E2E: Deploy hook error handling."""

from __future__ import annotations

import pytest

from synix.mesh.deploy import run_deploy_hooks


class TestDeployHookFailure:
    def test_nonexistent_command_raises(self, tmp_path):
        """Hook with nonexistent command raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Deploy hook failed"):
            run_deploy_hooks(
                ["/usr/bin/false"],
                build_dir=tmp_path,
            )

    def test_build_dir_substitution_in_hook(self, tmp_path):
        """Hook command receives {build_dir} substitution."""
        marker = tmp_path / "hook-ran"
        run_deploy_hooks(
            ["touch {build_dir}/hook-marker"],
            build_dir=tmp_path,
        )
        assert (tmp_path / "hook-marker").exists()

    def test_successful_hook_does_not_raise(self, tmp_path):
        """Successful hook completes without error."""
        run_deploy_hooks(["/usr/bin/true"], build_dir=tmp_path)

    def test_multiple_hooks_stop_on_failure(self, tmp_path):
        """When a hook fails, subsequent hooks don't run."""
        marker = tmp_path / "should-not-exist"
        with pytest.raises(RuntimeError):
            run_deploy_hooks(
                ["/usr/bin/false", f"touch {marker}"],
                build_dir=tmp_path,
            )
        assert not marker.exists()
