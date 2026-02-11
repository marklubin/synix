"""LLMs command — synix llms."""

from __future__ import annotations

import importlib.resources
import sys
from pathlib import Path

import click

from synix.cli.main import console


def _get_bundled_llms_txt() -> str | None:
    """Read the llms.txt bundled with the package."""
    try:
        ref = importlib.resources.files("synix").joinpath("llms.txt")
        return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        return None


def _collect_docs() -> str:
    """Collect all documentation files from the project for the LLM to review."""
    project_root = Path.cwd()
    docs = []

    # CLAUDE.md
    claude_md = project_root / "CLAUDE.md"
    if claude_md.exists():
        docs.append(f"# File: CLAUDE.md\n\n{claude_md.read_text()}")

    # README.md
    readme = project_root / "README.md"
    if readme.exists():
        docs.append(f"# File: README.md\n\n{readme.read_text()}")

    # docs/ directory
    docs_dir = project_root / "docs"
    if docs_dir.is_dir():
        for md_file in sorted(docs_dir.glob("*.md")):
            docs.append(f"# File: docs/{md_file.name}\n\n{md_file.read_text()}")

    # pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        docs.append(f"# File: pyproject.toml\n\n{pyproject.read_text()}")

    if not docs:
        return ""

    return "\n\n---\n\n".join(docs)


def _generate_llms_txt() -> str:
    """Call Claude to generate llms.txt from project documentation."""
    import anthropic

    docs_content = _collect_docs()
    if not docs_content:
        console.print("[red]Error:[/red] No documentation files found in current directory.")
        sys.exit(1)

    client = anthropic.Anthropic()

    prompt = f"""You are reviewing a software project's documentation to produce an llms.txt file.

llms.txt is a standard format that helps LLMs understand a project quickly. It should be a concise,
structured plain-text document that covers:

1. **Project name and one-line description**
2. **What it does** — core functionality in 2-3 sentences
3. **Key concepts** — the domain model / mental model needed to use it
4. **Installation and quick start** — how to get running
5. **CLI commands** — available commands with brief descriptions
6. **Architecture overview** — module structure, key files
7. **Pipeline definition** — how users define their pipelines
8. **Important constraints** — things to know (e.g., SQLite only, no web UI)

Keep it under 200 lines. Be factual and specific. No marketing language. Write it so an LLM
agent could read this file and immediately start working with the project.

Use this format:
```
# <project-name>

> <one-line description>

## <section>
<content>
```

Here is the project documentation:

<documentation>
{docs_content}
</documentation>

Generate the llms.txt content now."""

    with console.status("[bold blue]Generating llms.txt from documentation..."):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

    return response.content[0].text


@click.command()
@click.option("--generate", is_flag=True, help="Generate llms.txt from project docs using Claude.")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for generated llms.txt (default: src/synix/llms.txt).",
)
def llms(generate: bool, output: str | None):
    """Display or generate the llms.txt project summary for LLMs."""
    if generate:
        content = _generate_llms_txt()

        if output:
            out_path = Path(output)
        else:
            # Default: write into the package source so it ships with the wheel
            out_path = Path(__file__).parent.parent / "llms.txt"

        out_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Generated llms.txt[/green] → {out_path}")
        console.print("[dim]Commit this file so it ships with the package.[/dim]")
        return

    # Display mode — read from package data
    content = _get_bundled_llms_txt()
    if content is None:
        console.print(
            "[yellow]No llms.txt found.[/yellow] Run [bold]synix llms --generate[/bold] "
            "in the project root to create one."
        )
        sys.exit(1)

    click.echo(content)
