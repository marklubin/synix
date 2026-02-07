"""DAG resolution — determine build order, detect what needs rebuild."""

from __future__ import annotations

from collections import deque

from synix import Layer, Pipeline


def resolve_build_order(pipeline: Pipeline) -> list[Layer]:
    """Topological sort of layers — return in build order."""
    layer_map = {layer.name: layer for layer in pipeline.layers}

    # Build in-degree and adjacency
    in_degree: dict[str, int] = {name: 0 for name in layer_map}
    children: dict[str, list[str]] = {name: [] for name in layer_map}

    for layer in pipeline.layers:
        for dep in layer.depends_on:
            if dep not in layer_map:
                raise ValueError(f"Layer '{layer.name}' depends on unknown layer '{dep}'")
            children[dep].append(layer.name)
            in_degree[layer.name] += 1

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
        raise ValueError(f"Circular dependency detected involving: {sorted(remaining)}")

    return [layer_map[name] for name in order]


def needs_rebuild(
    artifact_id: str,
    current_input_hashes: list[str],
    current_prompt_id: str | None,
    store,
) -> bool:
    """Check if an artifact needs to be rebuilt."""
    existing = store.load_artifact(artifact_id)
    if existing is None:
        return True
    if sorted(existing.input_hashes) != sorted(current_input_hashes):
        return True
    if existing.prompt_id != current_prompt_id:
        return True
    return False
