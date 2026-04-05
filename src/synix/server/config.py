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
class BuildQueueConfig:
    """Event-driven build queue settings.

    First new document starts an N-second window timer.  All documents
    arriving within the window form one batch.  Documents arriving after
    are blocked until the current build releases its lock.
    """

    enabled: bool = True
    window: int = 30  # seconds — batch window after first enqueue


@dataclass
class VLLMConfig:
    """vLLM subprocess configuration."""

    enabled: bool = False
    model: str = "Qwen/Qwen3.5-2B"
    gpu_device: int = 0
    port: int = 8100
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 120


@dataclass
class ServerConfig:
    """Top-level server configuration."""

    project_dir: str
    pipeline_path: str = "pipeline.py"
    mcp_port: int = 8200
    viewer_port: int = 9471
    viewer_host: str = "0.0.0.0"
    buckets: list[BucketConfig] = field(default_factory=list)
    auto_build: BuildQueueConfig = field(default_factory=BuildQueueConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
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

    # Build queue (backward-compatible with old auto_build keys)
    auto_build_raw = raw.get("auto_build", {})
    auto_build = BuildQueueConfig(
        enabled=auto_build_raw.get("enabled", True),
        window=int(auto_build_raw.get("window", 30)),
    )

    # vLLM
    vllm_raw = raw.get("vllm", {})
    vllm = VLLMConfig(
        enabled=vllm_raw.get("enabled", False),
        model=vllm_raw.get("model", "Qwen/Qwen3.5-2B"),
        gpu_device=int(vllm_raw.get("gpu_device", 0)),
        port=int(vllm_raw.get("port", 8100)),
        max_model_len=int(vllm_raw.get("max_model_len", 2048)),
        gpu_memory_utilization=float(vllm_raw.get("gpu_memory_utilization", 0.85)),
        extra_args=vllm_raw.get("extra_args", []),
        startup_timeout=int(vllm_raw.get("startup_timeout", 120)),
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
        vllm=vllm,
        allowed_hosts=allowed_hosts,
    )
