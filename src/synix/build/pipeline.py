"""Parse pipeline Python config into Pipeline/Layer objects."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from synix.build.dag import resolve_build_order
from synix.core.models import Pipeline


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

    # Check at least one level-0 layer (multiple roots are allowed for multi-source pipelines)
    root_layers = [l for l in pipeline.layers if l.level == 0]
    if len(root_layers) == 0:
        raise ValueError("Pipeline must have at least one level-0 (source) layer, found none")

    # Check DAG is acyclic — resolve_build_order raises ValueError on cycles
    resolve_build_order(pipeline)
