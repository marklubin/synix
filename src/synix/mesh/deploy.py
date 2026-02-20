"""Deploy hook runner — executes shell commands with build_dir substitution."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

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
        logger.info("Running deploy hook: %s", rendered)
        result = subprocess.run(
            rendered,
            shell=True,
            cwd=build_dir,
            timeout=DEPLOY_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "Deploy hook failed: cmd=%r rc=%d stderr=%s",
                rendered,
                result.returncode,
                result.stderr,
            )
            raise RuntimeError(f"Deploy hook failed (rc={result.returncode}): {rendered}")
        logger.debug("Deploy hook succeeded: %s", rendered)
