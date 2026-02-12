"""Base transform interface and registry."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from synix.core.models import Artifact

PROMPTS_DIR = Path(__file__).parent / "prompts"


class BaseTransform(ABC):
    """Abstract base class for all transforms."""

    @abstractmethod
    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """Transform input artifacts into output artifacts.

        Returns a list because some transforms produce multiple outputs
        (e.g., one episode per conversation, one rollup per month).
        """
        ...

    def load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts/ directory."""
        path = PROMPTS_DIR / f"{name}.txt"
        return path.read_text()

    def get_prompt_id(self, template_name: str) -> str:
        """Generate a versioned prompt ID from the template content hash."""
        content = self.load_prompt(template_name)
        hash_prefix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{template_name}_v{hash_prefix}"

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        """Split inputs into independently-processable work units.

        Each unit is (unit_inputs, config_extras). The runner calls split()
        to determine parallelism, then executes each unit (potentially
        concurrently) via execute(unit_inputs, {**config, **config_extras}).

        Default: 1:1 — one unit per input artifact. When inputs is empty
        (e.g., source/parse transforms), returns a single unit so execute()
        is still called. Override for transforms that need different
        decomposition (e.g., N:1 for core synthesis, group-by-month for
        monthly rollup).
        """
        if not inputs:
            return [(inputs, {})]
        return [([inp], {}) for inp in inputs]

    def get_cache_key(self, config: dict) -> str:
        """Return a hash of transform-specific config that affects output.

        Override in subclasses to include config values that should
        invalidate the cache when changed (e.g., topics list, context_budget).
        Default returns empty string (no extra cache key).
        """
        return ""


# Transform registry
_TRANSFORMS: dict[str, type[BaseTransform]] = {}


def register_transform(name: str):
    """Decorator to register a transform class."""

    def wrapper(cls):
        _TRANSFORMS[name] = cls
        return cls

    return wrapper


def get_transform(name: str) -> BaseTransform:
    """Get an instantiated transform by name."""
    if name not in _TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}. Available: {list(_TRANSFORMS.keys())}")
    return _TRANSFORMS[name]()
