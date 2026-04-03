"""Server configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BucketConfig:
    """A single ingest bucket."""

    name: str
    dir: str
    patterns: list[str] = field(default_factory=lambda: ["**/*"])
    description: str = ""


@dataclass
class AutoBuildConfig:
    """Auto-build settings for the knowledge server."""

    enabled: bool = True
    scan_interval: int = 60  # seconds between source scans
    cooldown: int = 300  # seconds between builds


@dataclass
class ServerConfig:
    """Top-level server configuration."""

    project_dir: str
    pipeline_path: str = "pipeline.py"
    mcp_port: int = 8200
    viewer_port: int = 9471
    viewer_host: str = "0.0.0.0"
    buckets: list[BucketConfig] = field(default_factory=list)
    auto_build: AutoBuildConfig = field(default_factory=AutoBuildConfig)
    allowed_hosts: list[str] = field(default_factory=list)


def load_config(path: str) -> ServerConfig:
    """Load server config from TOML file.

    Raises FileNotFoundError if path does not exist.
    Raises KeyError if required fields are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    return _parse_config(raw)


def _parse_config(raw: dict) -> ServerConfig:
    """Parse raw TOML dict into ServerConfig."""
    server_raw = raw.get("server", {})

    if "project_dir" not in server_raw:
        raise KeyError("server.project_dir is required in config")

    # Buckets
    buckets = []
    for name, bucket_raw in raw.get("buckets", {}).items():
        buckets.append(
            BucketConfig(
                name=name,
                dir=bucket_raw.get("dir", f"./{name}"),
                patterns=bucket_raw.get("patterns", ["**/*"]),
                description=bucket_raw.get("description", ""),
            )
        )

    # Auto-build
    auto_build_raw = raw.get("auto_build", {})
    auto_build = AutoBuildConfig(
        enabled=auto_build_raw.get("enabled", True),
        scan_interval=int(auto_build_raw.get("scan_interval", 60)),
        cooldown=int(auto_build_raw.get("cooldown", 300)),
    )

    # Allowed hosts
    allowed_hosts = server_raw.get("allowed_hosts", [])

    return ServerConfig(
        project_dir=server_raw["project_dir"],
        pipeline_path=server_raw.get("pipeline_path", "pipeline.py"),
        mcp_port=int(server_raw.get("mcp_port", 8200)),
        viewer_port=int(server_raw.get("viewer_port", 9471)),
        viewer_host=server_raw.get("viewer_host", "0.0.0.0"),
        buckets=buckets,
        auto_build=auto_build,
        allowed_hosts=allowed_hosts,
    )
