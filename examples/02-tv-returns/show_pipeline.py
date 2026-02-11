"""Display the pipeline definition with syntax highlighting."""
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

source = Path("pipeline.py").read_text()

# Show only the pipeline definition section (skip transform implementations)
lines = source.splitlines()
start = next(i for i, l in enumerate(lines) if "PIPELINE DEFINITION" in l) - 1
definition = "\n".join(lines[start:])

console = Console(width=110)
console.print(Syntax(definition, "python", theme="monokai", line_numbers=True,
                     start_line=start + 1))
