"""Parse pipeline Python config into Pipeline/Layer objects."""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path

from synix import Pipeline


def load_pipeline(path: str) -> Pipeline:
    """Import a Python pipeline module and extract the `pipeline` variable."""
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    module_name = f"_synix_pipeline_{filepath.stem}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    pipeline = getattr(module, "pipeline", None)
    if pipeline is None:
        raise ValueError(f"Pipeline module {path} must define a 'pipeline' variable")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"'pipeline' variable must be a Pipeline instance, got {type(pipeline)}")

    validate_pipeline(pipeline)
    return pipeline


def validate_pipeline(pipeline: Pipeline) -> None:
    """Validate pipeline configuration."""
    if not pipeline.layers:
        raise ValueError("Pipeline must have at least one layer")

    layer_names = {layer.name for layer in pipeline.layers}

    # Check all depends_on reference existing layer names
    for layer in pipeline.layers:
        for dep in layer.depends_on:
            if dep not in layer_names:
                raise ValueError(
                    f"Layer '{layer.name}' depends on '{dep}', which does not exist. "
                    f"Available layers: {sorted(layer_names)}"
                )

    # Check exactly one level-0 layer
    root_layers = [l for l in pipeline.layers if l.level == 0]
    if len(root_layers) == 0:
        raise ValueError("Pipeline must have exactly one level-0 (source) layer, found none")
    if len(root_layers) > 1:
        names = [l.name for l in root_layers]
        raise ValueError(
            f"Pipeline must have exactly one level-0 (source) layer, found {len(root_layers)}: {names}"
        )

    # Check DAG is acyclic via topological sort
    _topological_sort(pipeline)


def _topological_sort(pipeline: Pipeline) -> list[str]:
    """Kahn's algorithm — raises ValueError on cycles."""
    layer_map = {layer.name: layer for layer in pipeline.layers}

    # Build adjacency and in-degree
    in_degree: dict[str, int] = {name: 0 for name in layer_map}
    children: dict[str, list[str]] = {name: [] for name in layer_map}

    for layer in pipeline.layers:
        for dep in layer.depends_on:
            children[dep].append(layer.name)
            in_degree[layer.name] += 1

    # Start with nodes that have no dependencies
    queue: deque[str] = deque()
    for name, degree in in_degree.items():
        if degree == 0:
            queue.append(name)

    order: list[str] = []
    while queue:
        name = queue.popleft()
        order.append(name)
        for child in children[name]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(layer_map):
        remaining = set(layer_map.keys()) - set(order)
        raise ValueError(f"Pipeline has circular dependencies involving: {sorted(remaining)}")

    return order
