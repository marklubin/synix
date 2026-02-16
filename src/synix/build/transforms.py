"""Transform infrastructure — prompt directory and backward compat aliases."""

from __future__ import annotations

from pathlib import Path

from synix.core.models import Transform  # noqa: F401

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Backward compatibility alias — new code should use Transform directly.
BaseTransform = Transform
