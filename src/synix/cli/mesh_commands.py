"""CLI commands for synix mesh — distributed build and deploy."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def mesh():
    """[Experimental] Manage synix meshes for distributed builds."""
    pass


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
@click.option("--pipeline", required=True, type=click.Path(exists=True), help="Path to pipeline.py")
def create(name: str, pipeline: str):
    """Create a new mesh (generates config and token)."""
    from synix.mesh.config import resolve_mesh_root
    from synix.mesh.provision import create_mesh

    try:
        config = create_mesh(name, Path(pipeline), mesh_root=resolve_mesh_root())
        console.print(f"[green]Created mesh '[bold]{name}[/bold]'[/green]")
        console.print(f"  Config: {config.mesh_dir / 'synix-mesh.toml'}")
        console.print(f"  Token:  {config.token[:12]}...")
        console.print()
        console.print("Next steps:")
        console.print(f"  uvx synix mesh provision --name {name} --role server")
        console.print(f"  uvx synix mesh provision --name {name} --role client --server HOST:PORT")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
@click.option("--role", required=True, type=click.Choice(["server", "client"]), help="Role for this machine")
@click.option("--server", "server_url", default="", help="Server address for client role (e.g., obispo:7433)")
def provision(name: str, role: str, server_url: str):
    """Provision this machine as server or client in a mesh."""
    from synix.mesh.config import resolve_mesh_root
    from synix.mesh.provision import provision_role

    if role == "client" and not server_url:
        console.print("[red]Error:[/red] --server is required for client role")
        sys.exit(1)

    # Normalize server URL
    if server_url and not server_url.startswith("http"):
        server_url = f"http://{server_url}"

    try:
        provision_role(name, role, server_url=server_url, mesh_root=resolve_mesh_root())
        console.print(f"[green]Provisioned as [bold]{role}[/bold] for mesh '[bold]{name}[/bold]'[/green]")
        if role == "server":
            console.print("  Server will start automatically via systemd")
        else:
            console.print(f"  Connected to server at {server_url}")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@mesh.command("server")
@click.option("--name", required=True, help="Mesh name")
def run_server(name: str):
    """Start the mesh server daemon."""
    import uvicorn

    from synix.mesh.config import load_mesh_config, resolve_mesh_root
    from synix.mesh.logging import setup_mesh_logging
    from synix.mesh.server import create_app

    config_path = resolve_mesh_root() / name / "synix-mesh.toml"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    config = load_mesh_config(config_path)
    setup_mesh_logging(
        config.mesh_dir,
        "server",
        file_level=config.logging_config.get_file_level(),
        stderr_level=config.logging_config.get_stderr_level(),
    )
    app = create_app(config)

    console.print(f"[green]Starting mesh server '[bold]{name}[/bold]' on port {config.server.port}[/green]")
    uvicorn.run(app, host="0.0.0.0", port=config.server.port, log_level="info")


@mesh.command("client")
@click.option("--name", required=True, help="Mesh name")
def run_client(name: str):
    """Start the mesh client daemon."""
    from synix.mesh.client import MeshClient
    from synix.mesh.config import load_mesh_config, resolve_mesh_root
    from synix.mesh.logging import setup_mesh_logging

    config_path = resolve_mesh_root() / name / "synix-mesh.toml"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    config = load_mesh_config(config_path)
    setup_mesh_logging(
        config.mesh_dir,
        "client",
        file_level=config.logging_config.get_file_level(),
        stderr_level=config.logging_config.get_stderr_level(),
    )
    client = MeshClient(config)

    console.print(f"[green]Starting mesh client '[bold]{name}[/bold]'[/green]")
    console.print(f"  Server: {client.server_url}")
    console.print(f"  Watch dir: {config.source.watch_dir}")

    asyncio.run(client.start())


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
def status(name: str):
    """Show mesh health, members, and last build info."""
    from synix.mesh.config import resolve_mesh_root

    mesh_dir = resolve_mesh_root() / name
    if not mesh_dir.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    state_path = mesh_dir / "state.json"
    if not state_path.exists():
        console.print(f"[yellow]Mesh '{name}' exists but has no state (not provisioned)[/yellow]")
        return

    state = json.loads(state_path.read_text())

    console.print(f"[bold]Mesh: {name}[/bold]")
    console.print(f"  Role:       {state.get('role', 'unknown')}")
    console.print(f"  Hostname:   {state.get('my_hostname', 'unknown')}")
    console.print(f"  Server URL: {state.get('server_url', 'none')}")
    console.print(f"  Term:       {state.get('term', {}).get('counter', 0)}")
    console.print(f"  Leader:     {state.get('term', {}).get('leader_id', 'none')}")
    console.print(f"  Config hash: {state.get('config_hash', 'none')[:16]}...")

    # If we're the server, try to query the status endpoint locally
    if state.get("role") == "server":
        server_url = state.get("server_url", "")
        if server_url:
            try:
                import httpx

                from synix.mesh.auth import auth_headers
                from synix.mesh.config import load_mesh_config

                config = load_mesh_config(mesh_dir / "synix-mesh.toml")
                headers = auth_headers(config.token)
                resp = httpx.get(f"{server_url}/api/v1/status", headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    console.print()
                    console.print("[bold]Server Stats:[/bold]")
                    console.print(f"  Builds:     {data.get('build_count', 0)}")
                    sessions = data.get("sessions", {})
                    total, pending = sessions.get("total", 0), sessions.get("pending", 0)
                    console.print(f"  Sessions:   {total} total, {pending} pending")
                    console.print(f"  Members:    {', '.join(data.get('members', []))}")
                    console.print(f"  Uptime:     {data.get('uptime_seconds', 0):.0f}s")
            except Exception:
                console.print("  [dim](server not reachable)[/dim]")


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
@click.argument("file_path", type=click.Path(exists=True))
def submit(name: str, file_path: str):
    """One-shot source file submission to mesh server."""
    import base64
    import hashlib

    import httpx

    from synix.mesh.auth import auth_headers
    from synix.mesh.config import load_mesh_config, resolve_mesh_root

    config_path = resolve_mesh_root() / name / "synix-mesh.toml"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    config = load_mesh_config(config_path)

    state_path = resolve_mesh_root() / name / "state.json"
    server_url = ""
    if state_path.exists():
        state = json.loads(state_path.read_text())
        server_url = state.get("server_url", "")

    if not server_url:
        console.print("[red]Error:[/red] No server URL configured. Run provision first.")
        sys.exit(1)

    path = Path(file_path)
    content = path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    encoded = base64.b64encode(content).decode()

    payload = {
        "session_id": path.stem,
        "project_dir": "manual",
        "content": encoded,
        "sha256": sha256,
    }

    headers = auth_headers(config.token)
    try:
        resp = httpx.post(f"{server_url}/api/v1/sessions", json=payload, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            data = resp.json()
            status_text = "new" if data.get("new") else "duplicate"
            console.print(f"[green]Submitted {path.name} ({status_text})[/green]")
        else:
            console.print(f"[red]Submit failed:[/red] {resp.status_code} {resp.text}")
            sys.exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to server at {server_url}")
        sys.exit(1)


@mesh.command("build")
@click.option("--name", required=True, help="Mesh name")
def trigger_build(name: str):
    """Force trigger a build on the mesh server."""
    import httpx

    from synix.mesh.auth import auth_headers
    from synix.mesh.config import load_mesh_config, resolve_mesh_root

    config_path = resolve_mesh_root() / name / "synix-mesh.toml"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    config = load_mesh_config(config_path)
    state_path = resolve_mesh_root() / name / "state.json"
    server_url = ""
    if state_path.exists():
        state = json.loads(state_path.read_text())
        server_url = state.get("server_url", "")

    if not server_url:
        console.print("[red]Error:[/red] No server URL configured")
        sys.exit(1)

    headers = auth_headers(config.token)
    try:
        resp = httpx.post(f"{server_url}/api/v1/builds/trigger", headers=headers, timeout=30)
        data = resp.json()
        if resp.status_code == 200:
            console.print("[green]Build started[/green]")
        elif resp.status_code == 202:
            console.print("[yellow]Build already in progress — queued for next run[/yellow]")
        else:
            console.print(f"[red]Trigger failed:[/red] {resp.status_code} {data}")
            sys.exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to server at {server_url}")
        sys.exit(1)


@mesh.command("deploy")
@click.option("--name", required=True, help="Mesh name")
def force_deploy(name: str):
    """Force run local deploy hooks."""
    from synix.mesh.config import load_mesh_config, resolve_mesh_root
    from synix.mesh.deploy import run_deploy_hooks

    config_path = resolve_mesh_root() / name / "synix-mesh.toml"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    config = load_mesh_config(config_path)
    mesh_dir = resolve_mesh_root() / name
    state = json.loads((mesh_dir / "state.json").read_text()) if (mesh_dir / "state.json").exists() else {}
    role = state.get("role", "client")

    commands = config.deploy.server_commands if role == "server" else config.deploy.client_commands
    if not commands:
        console.print(f"[yellow]No deploy hooks configured for {role} role[/yellow]")
        return

    build_dir = mesh_dir / role / "build"
    if not build_dir.exists():
        console.print(f"[red]Error:[/red] No build directory at {build_dir}")
        sys.exit(1)

    try:
        run_deploy_hooks(commands, build_dir)
        console.print("[green]Deploy hooks completed successfully[/green]")
    except RuntimeError as exc:
        console.print(f"[red]Deploy failed:[/red] {exc}")
        sys.exit(1)


@mesh.command("list")
def list_cmd():
    """List all meshes on this machine."""
    from synix.mesh.config import resolve_mesh_root
    from synix.mesh.provision import list_meshes

    meshes = list_meshes(mesh_root=resolve_mesh_root())
    if not meshes:
        console.print("[dim]No meshes found[/dim]")
        return

    table = Table(title="Synix Meshes")
    table.add_column("Name", style="bold")
    table.add_column("Role")
    table.add_column("Server URL")
    table.add_column("Hostname")
    table.add_column("Path", style="dim")

    for m in meshes:
        table.add_row(
            m["name"],
            m.get("role", ""),
            m.get("server_url", ""),
            m.get("hostname", ""),
            m["path"],
        )

    console.print(table)


@mesh.command("rotate-token")
@click.option("--name", required=True, help="Mesh name")
def cmd_rotate_token(name: str):
    """Generate a new mesh token and update config."""
    from synix.mesh.config import resolve_mesh_root
    from synix.mesh.provision import rotate_token

    try:
        new_token = rotate_token(name, mesh_root=resolve_mesh_root())
        console.print(f"[green]Token rotated for mesh '[bold]{name}[/bold]'[/green]")
        console.print(f"  New token: {new_token[:12]}...")
        console.print()
        console.print("[yellow]Important:[/yellow] Re-provision all nodes with the new token:")
        console.print(f"  uvx synix mesh provision --name {name} --role server")
        console.print(f"  uvx synix mesh provision --name {name} --role client --server HOST:PORT")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
@click.option("--refresh", default=3, help="Refresh interval in seconds")
@click.option("--lines", default=20, help="Number of log lines to display")
def dashboard(name: str, refresh: int, lines: int):
    """Live dashboard showing mesh health, members, and activity."""
    from synix.mesh.config import load_mesh_config, resolve_mesh_root
    from synix.mesh.dashboard import run_dashboard

    mesh_dir = resolve_mesh_root() / name
    if not mesh_dir.exists():
        console.print(f"[red]Error:[/red] Mesh '{name}' not found")
        sys.exit(1)

    # Load config for token and server URL
    config_path = mesh_dir / "synix-mesh.toml"
    token = ""
    server_url = ""
    if config_path.exists():
        config = load_mesh_config(config_path)
        token = config.token

    state_path = mesh_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        server_url = state.get("server_url", "")

    run_dashboard(
        mesh_dir=mesh_dir,
        name=name,
        token=token,
        server_url=server_url,
        refresh_interval=refresh,
        log_lines=lines,
    )


@mesh.command()
@click.option("--name", required=True, help="Mesh name")
@click.confirmation_option(prompt="This will remove all mesh data. Continue?")
def destroy(name: str):
    """Remove mesh config, data, and stop services."""
    from synix.mesh.config import resolve_mesh_root
    from synix.mesh.provision import destroy_mesh

    destroy_mesh(name, mesh_root=resolve_mesh_root())
    console.print(f"[green]Destroyed mesh '[bold]{name}[/bold]'[/green]")
