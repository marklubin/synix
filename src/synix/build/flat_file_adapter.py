"""FlatFile adapter — renders artifacts as a markdown context document.

Atomic write via tempfile + ``os.replace`` ensures the output file is
never partially written.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from synix.build.adapters import AdapterReceipt, ProjectionAdapter, ReleasePlan

logger = logging.getLogger(__name__)


class FlatFileAdapter(ProjectionAdapter):
    """Renders artifacts as a markdown context document."""

    def plan(self, closure, declaration, current_receipt=None):
        input_labels = set(declaration.input_artifacts)
        artifacts = {label: art for label, art in closure.artifacts.items() if label in input_labels}

        action = "rebuild"
        summary = f"Render context document from {len(artifacts)} artifacts"

        plan = ReleasePlan(
            adapter="flat_file",
            projection_name=declaration.name,
            action=action,
            summary=summary,
            artifacts_count=len(artifacts),
        )
        plan.details["_closure"] = closure
        plan.details["_declaration"] = declaration
        return plan

    def apply(self, plan, target):
        target_path = Path(target)

        closure = plan.details.get("_closure")
        declaration = plan.details.get("_declaration")
        if closure is None or declaration is None:
            raise RuntimeError("plan.details must contain '_closure' and '_declaration'")

        input_labels = declaration.input_artifacts
        parts = []
        for label in input_labels:
            art = closure.artifacts.get(label)
            if art is not None:
                parts.append(art.content)

        content = "\n\n".join(parts)

        # Determine output path
        output_filename = declaration.config.get("output_path", "context.md")
        if target_path.is_dir():
            output_path = target_path / Path(output_filename).name
        else:
            output_path = target_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via tempfile + os.replace
        fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(output_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        return AdapterReceipt(
            adapter="flat_file",
            projection_name=plan.projection_name,
            target=str(output_path),
            artifacts_applied=len(parts),
            status="success",
        )

    def verify(self, receipt, target):
        target_path = Path(receipt.target)
        return target_path.exists()
