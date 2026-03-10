"""Unit tests for Layer.level property (Bug 8 — synix info crash fix)."""

from __future__ import annotations

from synix.build.dag import compute_levels
from synix.core.models import Artifact, Source, Transform, TransformContext


class _DummyTransform(Transform):
    """Minimal concrete transform for testing."""

    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        return inputs


class TestLayerLevelProperty:
    def test_layer_level_property_default(self):
        """Layer.level should default to 0."""
        layer = Source("test-source")
        assert layer.level == 0

    def test_layer_level_property_after_set(self):
        """Layer.level should reflect changes to _level."""
        layer = Source("test-source")
        assert layer.level == 0
        layer._level = 3
        assert layer.level == 3

    def test_layer_level_after_compute_levels(self):
        """compute_levels() should set correct levels on Source and Transform."""
        src = Source("src")
        t1 = _DummyTransform("t1", depends_on=[src])
        t2 = _DummyTransform("t2", depends_on=[t1])

        compute_levels([src, t1, t2])

        assert src.level == 0
        assert t1.level == 1
        assert t2.level == 2

    def test_layer_level_property_is_readonly(self):
        """Setting .level should raise AttributeError (read-only property)."""
        layer = Source("test-source")
        try:
            layer.level = 5
            assert False, "Expected AttributeError for read-only property"
        except AttributeError:
            pass
