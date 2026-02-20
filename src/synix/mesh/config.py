"""TOML config loading and mesh directory management."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default mesh root: ~/.synix-mesh/
DEFAULT_MESH_ROOT = Path.home() / ".synix-mesh"


@dataclass
class SourceConfig:
    watch_dir: str = "~/.claude/projects"
    patterns: list[str] = field(default_factory=lambda: ["**/*.jsonl"])
    exclude: list[str] = field(default_factory=list)
    env_var: str = "SYNIX_SOURCE_DIR"


@dataclass
class ServerConfig:
    port: int = 7433
    build_min_interval: int = 300
    build_batch_threshold: int = 5
    build_max_delay: int = 900


@dataclass
class ClientConfig:
    scan_interval: int = 30
    pull_interval: int = 60
    heartbeat_interval: int = 120


@dataclass
class ClusterConfig:
    leader_candidates: list[str] = field(default_factory=list)
    leader_timeout: int = 360

    @property
    def config_hash(self) -> str:
        """SHA-256 of ordered candidate list. Preserves insertion order."""
        payload = "\0".join(self.leader_candidates)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class BundleConfig:
    include: list[str] = field(
        default_factory=lambda: [
            "manifest.json",
            "status.json",
            "layer2-*/**",
            "layer3-*/**",
            "project-contexts/**",
            "master-context.md",
            "context.md",
        ]
    )
    exclude: list[str] = field(default_factory=lambda: ["search.db", "*.lock", "logs/**"])


@dataclass
class DeployConfig:
    server_commands: list[str] = field(default_factory=list)
    client_commands: list[str] = field(default_factory=list)


@dataclass
class NotificationConfig:
    webhook_url: str = ""
    source: str = ""


@dataclass
class MeshConfig:
    name: str
    token: str = ""
    pipeline_path: str = "./pipeline.py"
    source: SourceConfig = field(default_factory=SourceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    bundle: BundleConfig = field(default_factory=BundleConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    @property
    def mesh_dir(self) -> Path:
        """Root directory for this mesh's data."""
        root = Path(os.environ.get("SYNIX_MESH_ROOT", str(DEFAULT_MESH_ROOT)))
        return root / self.name


def _get_list(d: dict, key: str, default: list | None = None) -> list:
    """Get a list value from dict, returning default if missing."""
    val = d.get(key)
    if val is None:
        return default if default is not None else []
    return list(val)


def load_mesh_config(path: Path) -> MeshConfig:
    """Load a MeshConfig from a TOML file.

    Raises ValueError if required fields are missing.
    """
    import tomllib

    if not path.exists():
        raise ValueError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    mesh = data.get("mesh", {})
    name = mesh.get("name")
    if not name:
        raise ValueError("Missing required field: [mesh].name")

    token = mesh.get("token", "")

    pipeline = data.get("pipeline", {})
    pipeline_path = pipeline.get("path", "./pipeline.py")

    # Source config
    src = data.get("source", {})
    source = SourceConfig(
        watch_dir=src.get("watch_dir", SourceConfig.watch_dir),
        patterns=_get_list(src, "patterns", SourceConfig().patterns),
        exclude=_get_list(src, "exclude", []),
        env_var=src.get("env_var", SourceConfig.env_var),
    )

    # Server config
    srv = data.get("server", {})
    server = ServerConfig(
        port=srv.get("port", ServerConfig.port),
        build_min_interval=srv.get("build_min_interval", ServerConfig.build_min_interval),
        build_batch_threshold=srv.get("build_batch_threshold", ServerConfig.build_batch_threshold),
        build_max_delay=srv.get("build_max_delay", ServerConfig.build_max_delay),
    )

    # Client config
    cli = data.get("client", {})
    client = ClientConfig(
        scan_interval=cli.get("scan_interval", ClientConfig.scan_interval),
        pull_interval=cli.get("pull_interval", ClientConfig.pull_interval),
        heartbeat_interval=cli.get("heartbeat_interval", ClientConfig.heartbeat_interval),
    )

    # Cluster config
    clust = data.get("cluster", {})
    cluster = ClusterConfig(
        leader_candidates=_get_list(clust, "leader_candidates", []),
        leader_timeout=clust.get("leader_timeout", ClusterConfig.leader_timeout),
    )

    # Bundle config
    bun = data.get("bundle", {})
    bundle = BundleConfig(
        include=_get_list(bun, "include", BundleConfig().include),
        exclude=_get_list(bun, "exclude", BundleConfig().exclude),
    )

    # Deploy config
    dep_server = data.get("deploy", {}).get("server", {})
    dep_client = data.get("deploy", {}).get("client", {})
    deploy = DeployConfig(
        server_commands=_get_list(dep_server, "commands", []),
        client_commands=_get_list(dep_client, "commands", []),
    )

    # Notification config
    notif = data.get("notifications", {})
    notifications = NotificationConfig(
        webhook_url=notif.get("webhook_url", ""),
        source=notif.get("source", name),
    )

    # Env var overrides
    if env_port := os.environ.get("SYNIX_MESH_PORT"):
        server.port = int(env_port)
    if env_watch := os.environ.get("SYNIX_MESH_SOURCE_WATCH_DIR"):
        source.watch_dir = env_watch

    return MeshConfig(
        name=name,
        token=token,
        pipeline_path=pipeline_path,
        source=source,
        server=server,
        client=client,
        cluster=cluster,
        bundle=bundle,
        deploy=deploy,
        notifications=notifications,
    )


def ensure_mesh_dirs(config: MeshConfig, role: str) -> None:
    """Create the mesh directory structure for the given role."""
    mesh_dir = config.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if role == "server":
        (mesh_dir / "server" / "sessions").mkdir(parents=True, exist_ok=True)
        (mesh_dir / "server" / "build").mkdir(parents=True, exist_ok=True)
        (mesh_dir / "server" / "bundles").mkdir(parents=True, exist_ok=True)
    elif role == "client":
        (mesh_dir / "client" / "build").mkdir(parents=True, exist_ok=True)
