# pipeline.py — Chunked Search Pipeline
#
# DAG:
#   Level 0: articles [parse]        -> one artifact per source document
#   Level 1: chunks [chunk]          -> paragraph-level chunks (1:N, no LLM)
#
# This pipeline splits documents into paragraph-level chunks and makes them
# searchable. No LLM calls — Chunk is pure text processing. Each chunk
# tracks provenance back to its source document and carries metadata
# (source_label, chunk_index, chunk_total) for downstream grouping.
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix release HEAD --to local
#   uvx synix search 'encryption' --release local

from synix import Pipeline, SearchSurface, Source, SynixSearch
from synix.transforms import Chunk

# -- Pipeline definition -----------------------------------------------------

pipeline = Pipeline("chunked-search")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"

# Level 0 — parse source documents
articles = Source("articles", dir="./sources/articles")

# Level 1 — split into paragraph chunks (no LLM)
chunks = Chunk(
    "chunks",
    depends_on=[articles],
    separator="\n\n",
    artifact_type="chunk",
)

chunk_search = SearchSurface(
    "chunk-search",
    sources=[chunks],
    modes=["fulltext"],
)

pipeline.add(articles, chunks, chunk_search)
pipeline.add(SynixSearch("search", surface=chunk_search))
