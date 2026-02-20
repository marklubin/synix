"""Deploy hook runner — executes shell commands with build_dir substitution."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from pathlib import Path

from synix.mesh.logging import mesh_event

logger = logging.getLogger(__name__)

DEPLOY_TIMEOUT = 300  # seconds


def run_deploy_hooks(commands: list[str], build_dir: Path) -> None:
    """Run deploy hook commands sequentially.

    Each command has ``{build_dir}`` substituted with the actual path.
    Non-zero exit raises ``RuntimeError`` with the command and return code.
    Commands run with ``shell=True``, timeout of 300s, ``cwd=build_dir``.
    """
    for cmd in commands:
        rendered = cmd.replace("{build_dir}", shlex.quote(str(build_dir)))
        mesh_event(
            logger,
            logging.INFO,
            f"Deploy hook started: {rendered}",
            "deploy_started",
            {
                "command": rendered,
            },
        )
        start = time.time()
        result = subprocess.run(
            rendered,
            shell=True,
            cwd=build_dir,
            timeout=DEPLOY_TIMEOUT,
            capture_output=True,
            text=True,
        )
        duration = time.time() - start
        if result.returncode != 0:
            mesh_event(
                logger,
                logging.ERROR,
                f"Deploy hook failed: {rendered} (rc={result.returncode})",
                "deploy_failed",
                {"command": rendered, "returncode": result.returncode, "stderr": result.stderr[:500]},
            )
            raise RuntimeError(f"Deploy hook failed (rc={result.returncode}): {rendered}")
        mesh_event(
            logger,
            logging.INFO,
            f"Deploy hook completed: {rendered}",
            "deploy_completed",
            {
                "command": rendered,
                "duration": round(duration, 1),
            },
        )
