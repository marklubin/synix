"""E2E tests — verify SynixSearchAdapter generates embeddings at release time.

Tests use real fastembed (BAAI/bge-small-en-v1.5), real FTS5, real file I/O.
No LLM mocking needed — embeddings are local ONNX models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from synix.build.release_engine import execute_release
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _embedding_projections(*, with_embedding_config: bool = True) -> dict:
    """Build projection declarations for tests."""
    config: dict = {"modes": ["fulltext", "semantic"]}
    if with_embedding_config:
        config["embedding_config"] = {
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        }
    return {
        "memory-search": {
            "adapter": "synix_search",
            "input_artifacts": ["ep-1", "ep-2"],
            "config": config,
            "config_fingerprint": "sha256:test-emb",
        },
    }


def _make_artifacts() -> dict[str, list[Artifact]]:
    return {
        "episodes": [
            Artifact(
                label="ep-1",
                artifact_type="episode",
                content="Memory systems and agent architectures for modern AI.",
            ),
            Artifact(
                label="ep-2",
                artifact_type="episode",
                content="Release management and deployment pipelines for production.",
            ),
        ],
    }


class TestReleaseEmbeddings:
    """Verify adapter generates embeddings at release time."""

    def test_release_creates_embeddings_dir(self, tmp_path: Path):
        """embedding_config declared → execute_release → embeddings/ exists."""
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=_embedding_projections(),
        )
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        embeddings_dir = release_dir / "embeddings"
        assert embeddings_dir.exists(), "embeddings/ dir should be created"
        assert (embeddings_dir / "manifest.json").exists(), "manifest.json should exist"

    def test_manifest_config_matches(self, tmp_path: Path):
        """Manifest _config matches declared embedding config."""
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=_embedding_projections(),
        )
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        manifest = json.loads((release_dir / "embeddings" / "manifest.json").read_text())
        stored_config = manifest["_config"]
        assert stored_config["provider"] == "fastembed"
        assert stored_config["model"] == "BAAI/bge-small-en-v1.5"

    def test_bin_files_exist(self, tmp_path: Path):
        """Binary embedding files exist for indexed artifacts."""
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=_embedding_projections(),
        )
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        embeddings_dir = release_dir / "embeddings"
        bin_files = list(embeddings_dir.glob("*.bin"))
        assert len(bin_files) >= 2, f"Expected at least 2 .bin files, got {len(bin_files)}"

    def test_semantic_search_from_release(self, tmp_path: Path):
        """HybridRetriever with EmbeddingProvider from release dir returns results."""
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=_embedding_projections(),
        )
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        from synix.core.config import EmbeddingConfig
        from synix.search.embeddings import EmbeddingProvider
        from synix.search.indexer import SearchIndex
        from synix.search.retriever import HybridRetriever

        config = EmbeddingConfig.from_dict({"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"})
        provider = EmbeddingProvider(config, str(release_dir))
        index = SearchIndex(release_dir / "search.db")
        retriever = HybridRetriever(index, provider)
        results = retriever.query("agent memory", mode="semantic", top_k=5)
        assert len(results) > 0, "Semantic search should return results"

    def test_fail_closed_embedding_error(self, tmp_path: Path):
        """embedding_config declared + provider fails → release raises RuntimeError."""
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=_embedding_projections(),
        )

        with patch(
            "synix.search.embeddings.EmbeddingProvider.embed_batch",
            side_effect=RuntimeError("Embedding model not found"),
        ):
            with pytest.raises(RuntimeError, match="Embedding model not found"):
                execute_release(synix_dir, release_name="local")

    def test_no_config_no_embeddings(self, tmp_path: Path):
        """No embedding_config → release succeeds → no embeddings/ dir."""
        projections = {
            "memory-search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-2"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test-no-emb",
            },
        }
        synix_dir = create_test_snapshot(
            tmp_path,
            _make_artifacts(),
            projections=projections,
        )
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        assert (release_dir / "search.db").exists()
        assert not (release_dir / "embeddings").exists(), "No embeddings/ when no config"
