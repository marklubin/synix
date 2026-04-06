"""Workspace — the first-class unit of configuration in Synix.

One Workspace = one synix namespace with identity, pipeline, buckets,
.synix build state, releases, and optional runtime services.

Config types (BucketConfig, BuildQueueConfig, VLLMConfig) are defined
here. server/config.py re-exports them for backward compatibility.
"""

from __future__ import annotations

import enum
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config types (canonical definitions — server/config.py re-exports these)
# ---------------------------------------------------------------------------


@dataclass
class BucketConfig:
    """A single ingest bucket."""

    name: str
    dir: str
    patterns: list[str] = field(default_factory=lambda: ["**/*"])
    description: str = ""


@dataclass
class BuildQueueConfig:
    """Event-driven build queue settings."""

    enabled: bool = True
    window: int = 30  # seconds — batch window after first enqueue


@dataclass
class VLLMConfig:
    """vLLM subprocess configuration."""

    enabled: bool = False
    model: str = "Qwen/Qwen3.5-2B"
    gpu_device: int = 0
    port: int = 8100
    max_model_len: int = 131072
    gpu_memory_utilization: float = 0.90
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 300


@dataclass
class WorkspaceConfig:
    """Parsed workspace configuration from synix.toml."""

    name: str = ""
    pipeline_path: str = "pipeline.py"
    buckets: list[BucketConfig] = field(default_factory=list)
    auto_build: BuildQueueConfig = field(default_factory=BuildQueueConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass
class ServerBindings:
    """Server-specific process bindings (ports, hosts)."""

    mcp_port: int = 8200
    viewer_port: int = 9471
    viewer_host: str = "0.0.0.0"
    allowed_hosts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workspace state
# ---------------------------------------------------------------------------


class WorkspaceState(enum.Enum):
    """Computed lifecycle state of a workspace.

    Describes *capability* — what operations are possible — not recency.
    A workspace with an old release is still RELEASED because search/viewer
    can serve it.
    """

    FRESH = "fresh"
    CONFIGURED = "configured"
    BUILT = "built"
    RELEASED = "released"
    SERVING = "serving"


# ---------------------------------------------------------------------------
# Runtime services (replaces _state dict in mcp_tools.py)
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceRuntime:
    """Runtime service handles bound to a serving workspace."""

    queue: Any  # DocumentQueue
    prompt_store: Any  # PromptStore
    build_lock: Any  # asyncio.Lock
    vllm_manager: Any = None
    llm_config_override: dict | None = None


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """A synix workspace — one named project with its config, pipeline,
    buckets, build state, and optional runtime services.

    Wraps Project via composition. Does not contain build logic — that
    stays in Project, runner.py, and search/.
    """

    def __init__(
        self,
        project: Any,  # sdk.Project (avoid circular import)
        config: WorkspaceConfig | None = None,
    ) -> None:
        self._project = project
        self._config = config or WorkspaceConfig()
        self._runtime: WorkspaceRuntime | None = None

    # --- Identity ---

    @property
    def name(self) -> str:
        if self._config.name:
            return self._config.name
        return self._project.project_root.name

    @property
    def root(self) -> Path:
        return self._project.project_root

    @property
    def synix_dir(self) -> Path:
        return self._project.synix_dir

    # --- State ---

    @property
    def state(self) -> WorkspaceState:
        """Computed lifecycle state (never stale)."""
        if self._runtime is not None:
            return WorkspaceState.SERVING
        if self._has_releases():
            return WorkspaceState.RELEASED
        if self._has_snapshots():
            return WorkspaceState.BUILT
        if self._project.pipeline is not None or self._config.buckets:
            return WorkspaceState.CONFIGURED
        return WorkspaceState.FRESH

    # --- Delegation to Project ---

    @property
    def project(self) -> Any:
        """The underlying Project for direct SDK access."""
        return self._project

    @property
    def pipeline(self):
        return self._project.pipeline

    def load_pipeline(self, path: str | Path | None = None):
        """Load pipeline. Uses config.pipeline_path if no arg given."""
        if path is None and self._config.pipeline_path:
            path = self.root / self._config.pipeline_path
        return self._project.load_pipeline(path)

    def build(self, **kwargs):
        return self._project.build(**kwargs)

    def release_to(self, name: str, ref: str = "HEAD"):
        return self._project.release_to(name, ref)

    def release(self, name: str):
        return self._project.release(name)

    def releases(self) -> list[str]:
        return self._project.releases()

    # --- Configuration ---

    @property
    def config(self) -> WorkspaceConfig:
        return self._config

    @property
    def buckets(self) -> list[BucketConfig]:
        return self._config.buckets

    def bucket_dir(self, name: str) -> Path:
        """Resolve a bucket's directory to an absolute path."""
        for b in self._config.buckets:
            if b.name == name:
                p = Path(b.dir)
                if not p.is_absolute():
                    return self.root / p
                return p
        available = [b.name for b in self._config.buckets]
        raise ValueError(f"Bucket {name!r} not found. Available: {available}")

    # --- Runtime services ---

    @property
    def runtime(self) -> WorkspaceRuntime | None:
        return self._runtime

    def activate_runtime(self, **kwargs) -> WorkspaceRuntime:
        """Bind runtime services. Called by serve()."""
        self._runtime = WorkspaceRuntime(**kwargs)
        return self._runtime

    # --- Private helpers ---

    def _has_snapshots(self) -> bool:
        try:
            from synix.build.refs import RefStore

            ref_store = RefStore(self.synix_dir)
            oid = ref_store.read_ref("refs/heads/main")
            return oid is not None
        except Exception:
            return False

    def _has_releases(self) -> bool:
        releases_dir = self.synix_dir / "releases"
        if not releases_dir.exists():
            return False
        return any((d / "receipt.json").exists() for d in releases_dir.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def open_workspace(path: str | Path = ".", config_path: str | Path | None = None) -> Workspace:
    """Open an existing synix workspace.

    Discovers synix.toml in the project root. If no config file exists,
    returns a bare workspace (just Project, no buckets/vllm/auto_build).
    """
    from synix.sdk import open_project

    project = open_project(str(path))
    config = _load_config(project.project_root, config_path)
    return Workspace(project, config)


def init_workspace(path: str | Path, pipeline: Any = None) -> Workspace:
    """Create a new workspace with synix.toml scaffold.

    Creates .synix/, pipeline.py template, source directories,
    and a starter synix.toml.
    """
    from synix.sdk import init as sdk_init

    project = sdk_init(str(path), pipeline=pipeline)
    project_root = project.project_root

    # Create synix.toml if it doesn't exist
    toml_path = project_root / "synix.toml"
    if not toml_path.exists():
        name = project_root.name
        toml_path.write_text(
            f'[workspace]\nname = "{name}"\npipeline_path = "pipeline.py"\n'
        )
        logger.info("Created %s", toml_path)

    # Ensure source directories exist
    sources_dir = project_root / "sources"
    sources_dir.mkdir(exist_ok=True)

    config = _load_config(project_root)
    return Workspace(project, config)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(project_root: Path, explicit_path: str | Path | None = None) -> WorkspaceConfig | None:
    """Load workspace config from synix.toml."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        return _parse_toml(p, project_root)

    toml_path = project_root / "synix.toml"
    if toml_path.exists():
        return _parse_toml(toml_path, project_root)

    return None


def _parse_toml(path: Path, project_root: Path) -> WorkspaceConfig:
    """Parse a synix.toml file into WorkspaceConfig."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    ws_raw = raw.get("workspace", {})
    name = ws_raw.get("name", project_root.name)
    pipeline_path = ws_raw.get("pipeline_path", "pipeline.py")

    # Buckets
    buckets = []
    for bname, bucket_raw in raw.get("buckets", {}).items():
        buckets.append(
            BucketConfig(
                name=bname,
                dir=bucket_raw.get("dir", f"./{bname}"),
                patterns=bucket_raw.get("patterns", ["**/*"]),
                description=bucket_raw.get("description", ""),
            )
        )

    # Build queue
    auto_build_raw = raw.get("auto_build", {})
    auto_build = BuildQueueConfig(
        enabled=auto_build_raw.get("enabled", True),
        window=int(auto_build_raw.get("window", 30)),
    )

    # vLLM — only pass present fields, let dataclass defaults handle the rest
    vllm_raw = raw.get("vllm", {})
    vllm_kwargs: dict = {}
    _vllm_keys = ("enabled", "model", "gpu_device", "port", "max_model_len",
                  "gpu_memory_utilization", "extra_args", "startup_timeout")
    for key in _vllm_keys:
        if key in vllm_raw:
            val = vllm_raw[key]
            if key in ("gpu_device", "port", "max_model_len", "startup_timeout"):
                val = int(val)
            elif key == "gpu_memory_utilization":
                val = float(val)
            vllm_kwargs[key] = val
    vllm = VLLMConfig(**vllm_kwargs)

    return WorkspaceConfig(
        name=name,
        pipeline_path=pipeline_path,
        buckets=buckets,
        auto_build=auto_build,
        vllm=vllm,
    )


def load_server_bindings(path: str | Path) -> ServerBindings:
    """Parse only the [server] section from a config file."""
    p = Path(path)
    if not p.exists():
        return ServerBindings()

    with open(p, "rb") as f:
        raw = tomllib.load(f)

    server_raw = raw.get("server", {})
    return ServerBindings(
        mcp_port=int(server_raw.get("mcp_port", 8200)),
        viewer_port=int(server_raw.get("viewer_port", 9471)),
        viewer_host=server_raw.get("viewer_host", "0.0.0.0"),
        allowed_hosts=server_raw.get("allowed_hosts", []),
    )
