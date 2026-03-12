"""synix view — open the web viewer for a release."""

from __future__ import annotations

import click

from synix.cli.main import console, main


@main.command()
@click.argument("release_name", default="local")
@click.option("--port", default=9471, type=int, help="Port to serve on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--title", default="Synix Viewer", help="Title shown in the UI")
def view(release_name: str, port: int, host: str, title: str) -> None:
    """Open the web viewer for a release (experimental)."""
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error:[/red] Flask is required for the viewer. "
            "Install it with: [bold]pip install synix\\[viewer][/bold]"
        )
        raise SystemExit(1) from None

    from synix.sdk import open_project
    from synix.viewer import serve

    project = open_project(".")
    release = project.release(release_name)
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            "[yellow]Warning:[/yellow] Binding to a non-local address. "
            "The viewer uses Flask's development server, which is not "
            "suitable for production use."
        )
    console.print(f"Opening viewer for release [bold]{release_name}[/bold] on http://{host}:{port}")
    serve(release, host=host, port=port, title=title, project=project)
