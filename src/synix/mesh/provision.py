"""Machine provisioning — directory creation, config copy, systemd unit installation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from synix.mesh.auth import generate_token
from synix.mesh.config import MeshConfig

logger = logging.getLogger(__name__)

# Systemd unit templates
SERVER_UNIT_TEMPLATE = """\
[Unit]
Description=Synix Mesh Server (%i)
After=network-online.target

[Service]
Type=simple
ExecStart={exec_start} mesh server --name %i
Restart=on-failure
RestartSec=15
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-%h/.config/synix-mesh.env
WorkingDirectory=%h/.synix-mesh/%i

[Install]
WantedBy=default.target
"""

CLIENT_UNIT_TEMPLATE = """\
[Unit]
Description=Synix Mesh Client (%i)
After=network-online.target

[Service]
Type=simple
ExecStart={exec_start} mesh client --name %i
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-%h/.config/synix-mesh.env
WorkingDirectory=%h/.synix-mesh/%i

[Install]
WantedBy=default.target
"""


def _get_synix_exec_path() -> str:
    """Get the path to the synix executable."""
    import sys

    venv_synix = Path(sys.executable).parent / "synix"
    if venv_synix.exists():
        return str(venv_synix)
    return "synix"


def create_mesh(name: str, pipeline_path: Path, mesh_root: Path | None = None) -> MeshConfig:
    """Create a new mesh with generated config.

    Returns the created MeshConfig.
    """
    from synix.mesh.config import DEFAULT_MESH_ROOT

    root = mesh_root or DEFAULT_MESH_ROOT
    mesh_dir = root / name
    if mesh_dir.exists():
        raise ValueError(f"Mesh '{name}' already exists at {mesh_dir}")

    mesh_dir.mkdir(parents=True)

    # Generate token
    token = generate_token()

    # Copy pipeline file
    dest_pipeline = mesh_dir / "pipeline.py"
    shutil.copy2(pipeline_path, dest_pipeline)

    # Write default config
    config_path = mesh_dir / "synix-mesh.toml"
    config_path.write_text(f"""\
[mesh]
name = "{name}"
token = "{token}"

[pipeline]
path = "./pipeline.py"

[source]
watch_dir = "~/.claude/projects"
patterns = ["**/*.jsonl"]
exclude = ["**/subagents/**", "**/tool-results/**"]
env_var = "SYNIX_SOURCE_DIR"

[server]
port = 7433
build_min_interval = 300
build_batch_threshold = 5
build_max_delay = 900

[client]
scan_interval = 30
pull_interval = 60
heartbeat_interval = 120

[cluster]
leader_candidates = []
leader_timeout = 360

[bundle]
include = ["manifest.json", "status.json", "context.md"]
exclude = ["search.db", "*.lock", "logs/**"]

[deploy.server]
commands = []

[deploy.client]
commands = []

[notifications]
webhook_url = ""
source = "{name}"
""")

    # Write initial state
    state = {
        "term": {"counter": 0, "leader_id": ""},
        "candidates": [],
        "config_hash": "",
        "role": "",
        "server_url": "",
        "my_hostname": "",
    }
    (mesh_dir / "state.json").write_text(json.dumps(state, indent=2))

    logger.info("Created mesh '%s' at %s", name, mesh_dir)

    from synix.mesh.config import load_mesh_config

    return load_mesh_config(config_path)


def provision_role(name: str, role: str, server_url: str = "", mesh_root: Path | None = None) -> None:
    """Provision this machine for the given role in a mesh.

    Creates data directories, updates state.json, and installs systemd units.
    """
    import socket

    from synix.mesh.config import DEFAULT_MESH_ROOT, load_mesh_config

    root = mesh_root or DEFAULT_MESH_ROOT
    mesh_dir = root / name
    config_path = mesh_dir / "synix-mesh.toml"

    if not config_path.exists():
        raise ValueError(f"Mesh '{name}' not found (no config at {config_path})")

    config = load_mesh_config(config_path)

    # Create role directories (use mesh_dir directly, not config.mesh_dir which uses env/defaults)
    if role == "server":
        (mesh_dir / "server" / "sessions").mkdir(parents=True, exist_ok=True)
        (mesh_dir / "server" / "build").mkdir(parents=True, exist_ok=True)
        (mesh_dir / "server" / "bundles").mkdir(parents=True, exist_ok=True)
    elif role == "client":
        (mesh_dir / "client" / "build").mkdir(parents=True, exist_ok=True)

    # Update state.json
    hostname = socket.gethostname()
    state_path = mesh_dir / "state.json"
    state = {
        "term": {"counter": 0, "leader_id": hostname if role == "server" else ""},
        "candidates": config.cluster.leader_candidates,
        "config_hash": config.cluster.config_hash,
        "role": role,
        "server_url": server_url or (f"http://localhost:{config.server.port}" if role == "server" else ""),
        "my_hostname": hostname,
    }
    state_path.write_text(json.dumps(state, indent=2))

    # Install systemd unit
    _install_systemd_unit(name, role)

    logger.info("Provisioned %s as %s for mesh '%s'", hostname, role, name)


def _install_systemd_unit(name: str, role: str) -> None:
    """Install and enable systemd user unit for the given role."""
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    exec_start = _get_synix_exec_path()
    template = SERVER_UNIT_TEMPLATE if role == "server" else CLIENT_UNIT_TEMPLATE
    unit_content = template.format(exec_start=exec_start)

    unit_name = f"synix-mesh-{role}@.service"
    unit_path = systemd_dir / unit_name
    unit_path.write_text(unit_content)

    # Enable and start the instance
    instance = f"synix-mesh-{role}@{name}"
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", instance],
            check=True,
            capture_output=True,
            timeout=30,
        )
        logger.info("Enabled and started systemd unit: %s", instance)
    except FileNotFoundError:
        logger.warning("systemctl not found — skipping systemd unit activation")
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode() if exc.stderr else str(exc)
        logger.warning("Failed to enable systemd unit %s: %s", instance, err_msg)


def destroy_mesh(name: str, mesh_root: Path | None = None) -> None:
    """Remove mesh config, data, and stop services."""
    from synix.mesh.config import DEFAULT_MESH_ROOT

    root = mesh_root or DEFAULT_MESH_ROOT
    mesh_dir = root / name

    # Stop systemd units
    for role in ("server", "client"):
        instance = f"synix-mesh-{role}@{name}"
        try:
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", instance],
                capture_output=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
        logger.info("Destroyed mesh '%s' at %s", name, mesh_dir)
    else:
        logger.warning("Mesh '%s' not found at %s", name, mesh_dir)


def list_meshes(mesh_root: Path | None = None) -> list[dict]:
    """List all meshes on this machine."""
    from synix.mesh.config import DEFAULT_MESH_ROOT

    root = mesh_root or DEFAULT_MESH_ROOT
    if not root.exists():
        return []

    meshes = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "synix-mesh.toml").exists():
            state_path = entry / "state.json"
            state = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text())
                except Exception:
                    logger.warning("Failed to read state for mesh '%s'", entry.name, exc_info=True)
            meshes.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "role": state.get("role", ""),
                    "server_url": state.get("server_url", ""),
                    "hostname": state.get("my_hostname", ""),
                }
            )
    return meshes


def rotate_token(name: str, mesh_root: Path | None = None) -> str:
    """Generate a new token and update the mesh config."""
    from synix.mesh.config import DEFAULT_MESH_ROOT

    root = mesh_root or DEFAULT_MESH_ROOT
    config_path = root / name / "synix-mesh.toml"

    if not config_path.exists():
        raise ValueError(f"Mesh '{name}' not found")

    new_token = generate_token()
    content = config_path.read_text()

    # Replace token line in TOML
    import re

    new_content = re.sub(r'token\s*=\s*"[^"]*"', f'token = "{new_token}"', content)
    config_path.write_text(new_content)

    logger.info("Rotated token for mesh '%s'", name)
    return new_token
