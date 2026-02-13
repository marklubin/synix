"""DAG resolution — determine build order, detect what needs rebuild."""

from __future__ import annotations

from collections import deque

from synix.build.fingerprint import Fingerprint
from synix.core.models import Layer, Pipeline


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
        raise ValueError(f"Pipeline has circular dependencies involving: {sorted(remaining)}")

    return [layer_map[name] for name in order]


def needs_rebuild(
    artifact_id: str,
    current_input_hashes: list[str],
    store,
    current_build_fingerprint: Fingerprint | None = None,
) -> tuple[bool, list[str]]:
    """Check if an artifact needs to be rebuilt.

    Returns (needs_rebuild, reasons) where reasons is a list of human-readable
    strings explaining why a rebuild is needed.

    Uses build fingerprint comparison: the fingerprint encodes transform identity
    (source code, prompt, model, config) plus input hashes into a single digest.
    """
    existing = store.load_artifact(artifact_id)
    if existing is None:
        return (True, ["new artifact"])

    if current_build_fingerprint is not None:
        stored_fp_data = (existing.metadata or {}).get("build_fingerprint")
        stored_fp = Fingerprint.from_dict(stored_fp_data) if stored_fp_data else None

        if stored_fp is None:
            return (True, ["no stored fingerprint"])

        if current_build_fingerprint.matches(stored_fp):
            return (False, [])

        reasons = current_build_fingerprint.explain_diff(stored_fp)
        return (True, reasons)

    # No fingerprint provided — can only check input hashes
    if sorted(existing.input_hashes) != sorted(current_input_hashes):
        return (True, ["inputs changed"])
    return (False, [])
