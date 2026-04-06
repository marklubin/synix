"""Synix SDK — programmatic access to synix as an infrastructure layer.

Provides complete programmatic access to everything the CLI can do:
init projects, define pipelines, manage sources, build, release, search, inspect.
Wraps existing internals (never duplicates them).
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SDK_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class SdkError(Exception):
    """Base class for all SDK errors."""


class SynixNotFoundError(SdkError):
    """No .synix/ directory found."""


class ReleaseNotFoundError(SdkError):
    """Named release doesn't exist."""


class ArtifactNotFoundError(SdkError):
    """Artifact label not found in snapshot."""


class SearchNotAvailableError(SdkError):
    """No search.db available for this release/surface."""


class EmbeddingRequiredError(SdkError):
    """embedding_config declared but embeddings missing at query time."""


class ProjectionNotFoundError(SdkError):
    """Named projection doesn't exist in the manifest."""


class PipelineRequiredError(SdkError):
    """Operation needs a pipeline but none is available."""


# ---------------------------------------------------------------------------
# SDK dataclasses (frozen, read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SdkArtifact:
    """Read-only artifact for SDK consumers."""

    label: str
    artifact_type: str
    content: str
    artifact_id: str
    layer: str
    layer_level: int
    provenance: list[str]
    metadata: dict
    prompt_id: str | None = None
    agent_fingerprint: str | None = None
    model_config: dict | None = None

    @classmethod
    def _from_resolved(cls, art) -> SdkArtifact:
        """Construct from a ResolvedArtifact."""
        return cls(
            label=art.label,
            artifact_type=art.artifact_type,
            content=art.content,
            artifact_id=art.artifact_id,
            layer=art.layer_name,
            layer_level=art.layer_level,
            provenance=list(art.provenance_chain),
            metadata=dict(art.metadata),
            prompt_id=getattr(art, "prompt_id", None),
            agent_fingerprint=getattr(art, "agent_fingerprint", None),
            model_config=getattr(art, "model_config", None),
        )


@dataclass(frozen=True)
class SdkSearchResult:
    """Read-only search result for SDK consumers."""

    content: str
    label: str
    layer: str
    layer_level: int
    score: float
    provenance: list[str]
    metadata: dict
    mode: str

    @classmethod
    def _from_internal(cls, r) -> SdkSearchResult:
        """Construct from a SearchResult."""
        return cls(
            content=r.content,
            label=r.label,
            layer=r.layer_name,
            layer_level=r.layer_level,
            score=r.score,
            provenance=list(r.provenance_chain),
            metadata=dict(r.metadata),
            mode=r.search_mode,
        )


@dataclass(frozen=True)
class BuildResult:
    """Summary of a pipeline build."""

    built: int
    cached: int
    skipped: int
    total_time: float
    snapshot_oid: str | None
    manifest_oid: str | None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_synix_dir(path: Path) -> Path:
    """Walk upward from path to find a .synix/ directory."""
    current = path.resolve()
    while True:
        candidate = current / ".synix"
        if candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            raise SynixNotFoundError(f"No .synix/ directory found at or above {path}")
        current = parent


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def open_project(path: str | Path = ".") -> Project:
    """Open an existing synix project. Walks upward to find .synix/."""
    p = Path(path).resolve()
    synix_dir = _discover_synix_dir(p)
    project_root = synix_dir.parent
    return Project(synix_dir, project_root)


def init(path: str | Path, *, pipeline=None) -> Project:
    """Create a new synix project at path.

    Creates .synix/ directory structure (objects/, refs/, HEAD).
    Optionally creates source directories from pipeline definition.
    """
    from synix.build.object_store import ObjectStore
    from synix.build.refs import RefStore

    project_root = Path(path).resolve()
    synix_dir = project_root / ".synix"

    # Create core structure
    ObjectStore(synix_dir)  # creates objects/ dir
    ref_store = RefStore(synix_dir)  # creates refs/ dir
    ref_store.ensure_head()  # creates HEAD

    project = Project(synix_dir, project_root)

    if pipeline is not None:
        project.set_pipeline(pipeline)
        # Create source directories from pipeline definition
        from synix.core.models import Source as SourceLayer

        source_dir = project_root / pipeline.source_dir
        source_dir.mkdir(parents=True, exist_ok=True)
        for layer in pipeline.layers:
            if isinstance(layer, SourceLayer):
                if layer.dir:
                    layer_dir = (project_root / layer.dir).resolve()
                else:
                    layer_dir = source_dir / layer.name
                layer_dir.mkdir(parents=True, exist_ok=True)

    return project


# ---------------------------------------------------------------------------
# SdkSource
# ---------------------------------------------------------------------------


class SdkSource:
    """Manage source files for a named pipeline source."""

    def __init__(self, source_dir: Path):
        self._dir = source_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _validate_name(self, name: str) -> Path:
        """Validate a filename and return its resolved path within the source dir."""
        resolved = (self._dir / name).resolve()
        if not resolved.parent == self._dir.resolve():
            raise SdkError(f"Invalid source file name {name!r}: must be a plain filename, not a path")
        return resolved

    def add(self, path: str | Path) -> None:
        """Copy a file into the source directory."""
        src = Path(path)
        dest = self._validate_name(src.name)
        shutil.copy2(str(src), str(dest))

    def add_text(self, content: str, label: str) -> None:
        """Create a text file in the source directory."""
        dest = self._validate_name(label)
        dest.write_text(content, encoding="utf-8")

    def list(self) -> list[str]:
        """List files in the source directory."""
        if not self._dir.exists():
            return []
        return sorted(f.name for f in self._dir.iterdir() if f.is_file())

    def remove(self, name: str) -> None:
        """Remove a source file."""
        target = self._validate_name(name)
        if target.exists():
            target.unlink()

    def clear(self) -> None:
        """Remove all source files."""
        if self._dir.exists():
            for f in self._dir.iterdir():
                if f.is_file():
                    f.unlink()


# ---------------------------------------------------------------------------
# SdkLayer
# ---------------------------------------------------------------------------


class SdkLayer:
    """Read-only layer info with artifact iteration."""

    def __init__(self, name: str, level: int, count: int, *, _release: Release):
        self.name = name
        self.level = level
        self.count = count
        self._release = _release

    def artifacts(self) -> Iterator[SdkArtifact]:
        """Yield all artifacts in this layer."""
        return self._release.artifacts(layer=self.name)


# ---------------------------------------------------------------------------
# SearchHandle
# ---------------------------------------------------------------------------


class SearchHandle:
    """Bound search client for a specific projection's search surface."""

    def __init__(self, release: Release, projection_name: str):
        self._release = release
        self._projection_name = projection_name

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        limit: int = 10,
        layers: list[str] | None = None,
    ) -> list[SdkSearchResult]:
        """Search within the bound surface."""
        return self._release.search(
            query,
            mode=mode,
            limit=limit,
            layers=layers,
            surface=self._projection_name,
        )


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


class Release:
    """Handle to a named or scratch release."""

    def __init__(self, synix_dir: Path, name: str):
        self._synix_dir = synix_dir
        self._name = name
        self._closure = None
        self._is_scratch = name == "HEAD"
        self._scratch_dir: Path | None = None
        self._scratch_release_name: str | None = None

    @property
    def name(self) -> str:
        return self._name

    def _release_dir(self) -> Path:
        """Return the directory where adapter outputs (search.db, etc.) live."""
        if self._is_scratch:
            if self._scratch_dir is None:
                scratch_id = uuid.uuid4().hex[:12]
                self._scratch_dir = self._synix_dir / "work" / f"scratch_{scratch_id}"
                self._scratch_dir.mkdir(parents=True, exist_ok=True)
                self._scratch_release_name = f"_scratch_{scratch_id}"
                # Materialize scratch release
                from synix.build.release_engine import execute_release

                execute_release(
                    self._synix_dir,
                    ref="HEAD",
                    release_name=self._scratch_release_name,
                    target=self._scratch_dir,
                )
            return self._scratch_dir
        return self._synix_dir / "releases" / self._name

    def _receipt_dir(self) -> Path:
        """Return the directory where execute_release writes the receipt."""
        if self._is_scratch and self._scratch_release_name:
            return self._synix_dir / "releases" / self._scratch_release_name
        return self._synix_dir / "releases" / self._name

    def _ensure_release_exists(self) -> Path:
        release_dir = self._release_dir()
        if not self._is_scratch and not (self._receipt_dir() / "receipt.json").exists():
            raise ReleaseNotFoundError(f"Release {self._name!r} not found at {release_dir}")
        return release_dir

    def _get_closure(self):
        if self._closure is None:
            from synix.build.release import ReleaseClosure

            self._ensure_release_exists()
            receipt_path = self._receipt_dir() / "receipt.json"

            # Read from receipt to ensure the closure matches the exact
            # snapshot that was materialized (avoids race if another build
            # lands between materialization and read).
            try:
                receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
                snapshot_oid = receipt_data["snapshot_oid"]
            except (KeyError, json.JSONDecodeError) as exc:
                raise ReleaseNotFoundError(f"Release {self._name!r} has a malformed receipt at {receipt_path}") from exc
            self._closure = ReleaseClosure.from_snapshot(self._synix_dir, snapshot_oid)
        return self._closure

    # -- Artifact access --

    def artifact(self, label: str) -> SdkArtifact:
        """Get a single artifact by label."""
        closure = self._get_closure()
        art = closure.artifacts.get(label)
        if art is None:
            raise ArtifactNotFoundError(f"Artifact {label!r} not found in release {self._name!r}")
        return SdkArtifact._from_resolved(art)

    def artifacts(self, layer: str | None = None) -> Iterator[SdkArtifact]:
        """Iterate all artifacts, optionally filtered by layer."""
        closure = self._get_closure()
        for art in closure.artifacts.values():
            if layer is not None and art.layer_name != layer:
                continue
            yield SdkArtifact._from_resolved(art)

    # -- Search --

    def _resolve_search_projection(self, surface: str | None = None):
        """Find the synix_search projection to use."""
        closure = self._get_closure()
        search_projections = {
            name: decl for name, decl in closure.projections.items() if decl.adapter == "synix_search"
        }

        if surface is not None:
            proj = search_projections.get(surface)
            if proj is None:
                available = list(search_projections.keys())
                raise SearchNotAvailableError(f"Search surface {surface!r} not found. Available: {available}")
            return proj

        if len(search_projections) == 1:
            return next(iter(search_projections.values()))

        if len(search_projections) == 0:
            raise SearchNotAvailableError("No search projections found in this release.")

        available = list(search_projections.keys())
        raise SearchNotAvailableError(f"Multiple search projections found: {available}. Specify surface= explicitly.")

    def _load_embedding_provider(self, projection):
        """Load embedding provider from release dir, or return None if no embeddings."""
        embedding_config = projection.config.get("embedding_config")
        if not embedding_config:
            return None

        from synix.core.config import EmbeddingConfig
        from synix.search.embeddings import EmbeddingProvider

        release_dir = self._ensure_release_exists()
        manifest_path = release_dir / "embeddings" / "manifest.json"
        if not manifest_path.exists():
            return None

        config = EmbeddingConfig.from_dict(embedding_config)
        return EmbeddingProvider(config, str(release_dir))

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        limit: int = 10,
        layers: list[str] | None = None,
        surface: str | None = None,
    ) -> list[SdkSearchResult]:
        """Search the release with fail-closed embedding enforcement."""
        projection = self._resolve_search_projection(surface)
        declared_modes = projection.config.get("modes", ["fulltext"])
        has_semantic = any(m in declared_modes for m in ("semantic", "hybrid", "layered"))

        # Fail-closed: if pipeline declares semantic capability, embeddings MUST exist
        if has_semantic and mode in ("semantic", "hybrid", "layered"):
            embedding_provider = self._load_embedding_provider(projection)
            if embedding_provider is None:
                raise EmbeddingRequiredError(
                    f"Surface declares modes={declared_modes} but no embeddings found "
                    f"in release directory. Re-run release to generate embeddings."
                )
        elif mode in ("semantic", "hybrid", "layered") and not has_semantic:
            raise SearchNotAvailableError(
                f"Requested mode={mode!r} but surface only declares modes={declared_modes}. "
                f"Add 'semantic' to SearchSurface modes to enable embedding search."
            )
        else:
            embedding_provider = None  # keyword-only is fine

        release_dir = self._ensure_release_exists()
        db_path = release_dir / "search.db"
        if not db_path.exists():
            raise SearchNotAvailableError(f"No search.db found in release {self._name!r}")

        from synix.cli.search_commands import ReleaseProvenanceProvider
        from synix.search.indexer import SearchIndex
        from synix.search.retriever import HybridRetriever

        search_index = SearchIndex(db_path)
        provenance = ReleaseProvenanceProvider(db_path)
        retriever = HybridRetriever(search_index, embedding_provider, provenance)
        results = retriever.query(query, mode=mode, layers=layers, top_k=limit)
        return [SdkSearchResult._from_internal(r) for r in results]

    def index(self, name: str) -> SearchHandle:
        """Get a bound search handle for a specific projection surface."""
        # Validate projection exists
        self._resolve_search_projection(name)
        return SearchHandle(self, name)

    # -- Inspect --

    def layers(self) -> list[SdkLayer]:
        """Return layer info for all layers in this release."""
        closure = self._get_closure()
        layer_info: dict[str, dict[str, Any]] = {}
        for art in closure.artifacts.values():
            name = art.layer_name
            if name not in layer_info:
                layer_info[name] = {"level": art.layer_level, "count": 0}
            layer_info[name]["count"] += 1

        return [
            SdkLayer(name, info["level"], info["count"], _release=self)
            for name, info in sorted(layer_info.items(), key=lambda x: x[1]["level"])
        ]

    def lineage(self, label: str) -> list[SdkArtifact]:
        """Walk provenance chain for an artifact."""
        closure = self._get_closure()
        art = closure.artifacts.get(label)
        if art is None:
            raise ArtifactNotFoundError(f"Artifact {label!r} not found in release {self._name!r}")

        # The provenance chain is already BFS-walked in ResolvedArtifact
        result = []
        for chain_label in art.provenance_chain:
            chain_art = closure.artifacts.get(chain_label)
            if chain_art is not None:
                result.append(SdkArtifact._from_resolved(chain_art))
        return result

    def _resolve_flat_file_path(self, name: str) -> Path:
        """Resolve a flat file projection to its output path on disk."""
        release_dir = self._ensure_release_exists()
        closure = self._get_closure()

        for proj_name, decl in closure.projections.items():
            if decl.adapter == "flat_file" and proj_name == name:
                output_path = decl.config.get("output_path", "context.md")
                file_path = release_dir / Path(output_path).name
                if file_path.exists():
                    return file_path
                raise ArtifactNotFoundError(
                    f"Flat file {output_path!r} not materialized at {file_path}. "
                    f"Run release to materialize projections."
                )

        available = [n for n, d in closure.projections.items() if d.adapter == "flat_file"]
        raise ProjectionNotFoundError(f"Flat file projection {name!r} not found. Available: {available}")

    def flat_file(self, name: str) -> str:
        """Return the content of a flat file projection."""
        return self._resolve_flat_file_path(name).read_text(encoding="utf-8")

    def flat_file_path(self, name: str) -> Path:
        """Return the path to a flat file projection."""
        return self._resolve_flat_file_path(name)

    def receipt(self) -> dict:
        """Return the release receipt as a dict."""
        self._ensure_release_exists()
        receipt_path = self._receipt_dir() / "receipt.json"
        if not receipt_path.exists():
            raise ReleaseNotFoundError(f"No receipt found for release {self._name!r}")
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    # -- Lifecycle --

    def close(self) -> None:
        """Clean up scratch release resources."""
        if self._is_scratch and self._scratch_dir is not None:
            if self._scratch_dir.exists():
                shutil.rmtree(self._scratch_dir)
            # Also clean up the receipt dir under releases/
            if self._scratch_release_name:
                receipt_dir = self._synix_dir / "releases" / self._scratch_release_name
                if receipt_dir.exists():
                    shutil.rmtree(receipt_dir)
            self._scratch_dir = None
            self._scratch_release_name = None
        self._closure = None

    def __enter__(self) -> Release:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class Project:
    """Handle to a synix project."""

    def __init__(self, synix_dir: Path, project_root: Path):
        self._synix_dir = synix_dir
        self._project_root = project_root
        self._pipeline = None

    @property
    def synix_dir(self) -> Path:
        return self._synix_dir

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def pipeline(self):
        return self._pipeline

    def set_pipeline(self, pipeline) -> None:
        """Attach a pipeline to this project."""
        self._pipeline = pipeline

    def load_pipeline(self, path: str | Path | None = None):
        """Load a pipeline from a Python file.

        If path is None, looks for pipeline.py in the project root.
        """
        from synix.build.pipeline import load_pipeline

        if path is None:
            default_path = self._project_root / "pipeline.py"
            if not default_path.exists():
                raise PipelineRequiredError(
                    f"No pipeline.py found in {self._project_root}. Pass a path or use set_pipeline()."
                )
            path = default_path

        pipeline = load_pipeline(str(path))
        self._pipeline = pipeline
        return pipeline

    def _resolve_pipeline(self, pipeline=None):
        """Resolve a pipeline from arguments, project state, or auto-detection."""
        if pipeline is not None:
            from synix.core.models import Pipeline

            if isinstance(pipeline, Pipeline):
                self._pipeline = pipeline
                return pipeline
            # Treat as file path
            return self.load_pipeline(pipeline)

        if self._pipeline is not None:
            return self._pipeline

        # Try auto-detect
        return self.load_pipeline()

    # -- Source management --

    def source(self, name: str) -> SdkSource:
        """Get a source manager for a named pipeline source.

        Raises PipelineRequiredError if no pipeline is set, or SdkError if the
        named source doesn't exist in the pipeline definition.
        """
        if self._pipeline is None:
            raise PipelineRequiredError(
                "Pipeline required for source management. Call set_pipeline() or build() first."
            )

        from synix.core.models import Source as SourceLayer

        # Find the Source layer by name
        for layer in self._pipeline.layers:
            if isinstance(layer, SourceLayer) and layer.name == name:
                if layer.dir:
                    source_dir = (self._project_root / layer.dir).resolve()
                else:
                    source_dir = self._project_root / self._pipeline.source_dir / name
                return SdkSource(source_dir)

        declared = [layer.name for layer in self._pipeline.layers if isinstance(layer, SourceLayer)]
        raise SdkError(f"Source {name!r} not declared in pipeline. Declared sources: {declared}")

    # -- Build & release --

    def build(
        self,
        pipeline=None,
        *,
        concurrency: int = 5,
        timeout: float | None = None,
        dry_run: bool = False,
        accept_existing: bool = False,
    ) -> BuildResult:
        """Build the pipeline and produce a snapshot.

        Args:
            pipeline: Optional pipeline override.
            concurrency: Max parallel transform workers.
            timeout: Per-request LLM timeout in seconds. Overrides pipeline llm_config.timeout.
            dry_run: If True, return plan counts without building.
            accept_existing: Keep cached artifacts even if the model/config changed.
                Only rebuild artifacts with new or changed inputs. Useful for
                model migration — existing work is preserved, new inputs use the
                new config.
        """
        original = self._resolve_pipeline(pipeline)

        # Deep-copy to avoid mutating the caller's pipeline object
        resolved = copy.deepcopy(original)

        # Apply timeout override to pipeline's LLM config
        if timeout is not None:
            llm_dict = dict(resolved.llm_config) if resolved.llm_config else {}
            llm_dict["timeout"] = timeout
            resolved.llm_config = llm_dict

        # Ensure synix_dir is set on the pipeline
        if resolved.synix_dir is None:
            resolved.synix_dir = str(self._synix_dir)

        # Make relative paths absolute relative to project root
        # (runner resolves them against CWD, which may not be the project)
        if resolved.source_dir and not Path(resolved.source_dir).is_absolute():
            resolved.source_dir = str((self._project_root / resolved.source_dir).resolve())
        if resolved.build_dir and not Path(resolved.build_dir).is_absolute():
            resolved.build_dir = str((self._project_root / resolved.build_dir).resolve())

        # Also resolve Source.dir on each layer
        from synix.core.models import Source as SourceLayer

        for layer in resolved.layers:
            if isinstance(layer, SourceLayer) and layer.dir and not Path(layer.dir).is_absolute():
                layer.dir = str((self._project_root / layer.dir).resolve())

        if dry_run:
            from synix.build.plan import plan_build

            plan_result = plan_build(resolved)
            return BuildResult(
                built=plan_result.total_rebuild,
                cached=plan_result.total_cached,
                skipped=0,
                total_time=0.0,
                snapshot_oid=None,
                manifest_oid=None,
            )

        from synix.build.runner import run

        result = run(resolved, concurrency=concurrency, accept_existing=accept_existing)
        return BuildResult(
            built=result.built,
            cached=result.cached,
            skipped=result.skipped,
            total_time=result.total_time,
            snapshot_oid=result.snapshot_oid,
            manifest_oid=result.manifest_oid,
        )

    def release_to(self, name: str, ref: str = "HEAD") -> dict:
        """Execute a release and return the receipt dict."""
        from synix.build.release_engine import execute_release

        receipt = execute_release(self._synix_dir, ref=ref, release_name=name)
        return receipt.to_dict()

    def release(self, name: str) -> Release:
        """Get a Release handle for a named release or HEAD."""
        return Release(self._synix_dir, name)

    # -- Inspect --

    def releases(self) -> list[str]:
        """List all release names."""
        from synix.build.release_engine import list_releases

        all_releases = list_releases(self._synix_dir)
        return [r["release_name"] for r in all_releases]

    def refs(self) -> dict[str, str]:
        """Return all refs as a flat dict."""
        from synix.build.refs import RefStore

        ref_store = RefStore(self._synix_dir)
        result: dict[str, str] = {}
        # Scan known ref prefixes
        for prefix in ("refs/heads", "refs/releases", "refs/runs"):
            try:
                result.update(ref_store.iter_refs(prefix))
            except (ValueError, FileNotFoundError):
                pass
        return result

    def clean(self) -> None:
        """Remove releases/ and work/ directories."""
        releases_dir = self._synix_dir / "releases"
        if releases_dir.exists():
            shutil.rmtree(releases_dir)

        work_dir = self._synix_dir / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
