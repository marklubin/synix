"""Integration test — Chunk transform in a full pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from synix import Pipeline, SearchSurface, Source, SynixSearch
from synix.build.release_engine import execute_release
from synix.build.runner import run
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.ext.chunk import Chunk


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create a source directory with plain text documents."""
    src = tmp_path / "sources" / "docs"
    src.mkdir(parents=True)
    (src / "readme.txt").write_text(
        "Synix is a build system for agent memory.\n\n"
        "It supports incremental rebuilds and full provenance tracking.\n\n"
        "Pipelines are declared in Python and produce searchable artifacts."
    )
    (src / "guide.txt").write_text(
        "Step 1: Install synix with uv.\n\nStep 2: Create a pipeline file.\n\nStep 3: Run synix build."
    )
    return src


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    return tmp_path / "build"


@pytest.fixture
def pipeline_with_chunks(build_dir: Path) -> Pipeline:
    p = Pipeline("chunk-test")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "test", "temperature": 0.0}

    docs = Source("docs")
    chunks = Chunk(
        "doc-chunks",
        depends_on=[docs],
        separator="\n\n",
        artifact_type="chunk",
    )
    surface = SearchSurface("chunk-surface", sources=[chunks], modes=["fulltext"])
    p.add(docs, chunks)
    p.add(surface)
    p.add(SynixSearch("search", surface=surface))
    return p


class TestChunkPipeline:
    def test_build_produces_chunks(self, pipeline_with_chunks, source_dir, build_dir):
        """Source → Chunk pipeline produces correct chunk artifacts."""
        result = run(pipeline_with_chunks, source_dir=str(source_dir.parent))

        synix_dir = build_dir.parent / ".synix"
        store = SnapshotArtifactCache(synix_dir)
        chunks = store.list_artifacts("doc-chunks")

        # readme.txt: 3 paragraphs, guide.txt: 3 paragraphs = 6 chunks
        assert len(chunks) == 6

        # All chunks have correct type
        for c in chunks:
            assert c.artifact_type == "chunk"

        # Check metadata on chunks
        for c in chunks:
            assert "source_label" in c.metadata
            assert "chunk_index" in c.metadata
            assert "chunk_total" in c.metadata

        # Layer stats
        layer_names = [s.name for s in result.layer_stats]
        assert "docs" in layer_names
        assert "doc-chunks" in layer_names

    def test_chunk_provenance(self, pipeline_with_chunks, source_dir, build_dir):
        """Each chunk traces back to its source document."""
        run(pipeline_with_chunks, source_dir=str(source_dir.parent))
        synix_dir = build_dir.parent / ".synix"
        store = SnapshotArtifactCache(synix_dir)

        chunks = store.list_artifacts("doc-chunks")
        docs = store.list_artifacts("docs")
        doc_ids = {d.artifact_id for d in docs}

        for chunk in chunks:
            # Every chunk's input_ids should point to a doc artifact
            assert len(chunk.input_ids) >= 1
            assert chunk.input_ids[0] in doc_ids

            # Parent labels should trace to source docs
            parents = store.get_parents(chunk.label)
            assert len(parents) >= 1

    def test_rebuild_chunks_cached(self, pipeline_with_chunks, source_dir, build_dir):
        """Second build with no changes caches all chunk artifacts."""
        run(pipeline_with_chunks, source_dir=str(source_dir.parent))
        result2 = run(pipeline_with_chunks, source_dir=str(source_dir.parent))

        # Sources always re-parse from disk, but chunks should be fully cached
        chunk_stats = [s for s in result2.layer_stats if s.name == "doc-chunks"]
        assert chunk_stats[0].cached == 6
        assert chunk_stats[0].built == 0

    def test_release_and_search(self, pipeline_with_chunks, source_dir, build_dir):
        """Build → release → search chunks works end-to-end."""
        run(pipeline_with_chunks, source_dir=str(source_dir.parent))
        synix_dir = build_dir.parent / ".synix"

        execute_release(synix_dir, ref="HEAD", release_name="local")

        release_dir = synix_dir / "releases" / "local"
        search_db = release_dir / "search.db"
        assert search_db.exists()

        from synix.search.indexer import SearchIndex as FTSIndex

        idx = FTSIndex(str(search_db))
        results = idx.query("incremental rebuilds")
        assert len(results) >= 1
        # Chunk metadata is stored in the search index
        assert any("source_label" in r.metadata for r in results)

    def test_chunk_config_change_invalidates_cache(self, source_dir, build_dir):
        """Changing chunk config triggers rebuild."""
        p1 = Pipeline("chunk-test")
        p1.build_dir = str(build_dir)
        p1.llm_config = {"model": "test", "temperature": 0.0}
        docs = Source("docs")
        chunks1 = Chunk("doc-chunks", depends_on=[docs], separator="\n\n")
        p1.add(docs, chunks1)

        run(p1, source_dir=str(source_dir.parent))

        # Change separator
        p2 = Pipeline("chunk-test")
        p2.build_dir = str(build_dir)
        p2.llm_config = {"model": "test", "temperature": 0.0}
        docs2 = Source("docs")
        chunks2 = Chunk("doc-chunks", depends_on=[docs2], separator="\n")
        p2.add(docs2, chunks2)

        result2 = run(p2, source_dir=str(source_dir.parent))
        # Chunks should be rebuilt (different separator)
        chunk_stats = [s for s in result2.layer_stats if s.name == "doc-chunks"]
        assert chunk_stats[0].built > 0
