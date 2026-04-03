<!-- created: 2026-04-03 -->
# Issue: fastembed hangs on large artifact sets (~23K chunks)

## Symptoms

- Search surface materialization with `modes=["fulltext", "semantic"]` and ~23,797 artifacts causes the build to hang indefinitely
- Process shows 0% CPU, ~1.2GB RSS — classic deadlock or blocked I/O
- No error, no crash, no timeout — just frozen
- FTS5 fulltext indexing for the same artifact set completes in seconds
- Happens specifically during the embedding generation phase in `_materialize_search_surface()`

## Environment

- salinas: AMD CPU, 120GB RAM, no GPU
- Python 3.13.12 (uv-managed)
- synix 0.22.2, fastembed 0.8.0, onnxruntime 1.24.4
- Model: BAAI/bge-small-en-v1.5 (384 dims)

## Reproduction

```python
# In a pipeline with ~300 reference docs:
reference_chunks = Chunk("ref-chunks", depends_on=[reference], chunk_size=512)
# Produces ~23,797 chunks

reference_surface = SearchSurface(
    "reference",
    sources=[reference_chunks],
    modes=["fulltext", "semantic"],  # THIS HANGS
    embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)
```

Changing to `modes=["fulltext"]` works fine. The hang is specifically in the semantic embedding path.

## Likely Causes (investigate)

1. **Batch size**: fastembed may try to embed all 23,797 chunks in one batch, exhausting memory or hitting ONNX thread pool limits
2. **ONNX thread deadlock**: onnxruntime 1.24.4 on Python 3.13 may have threading issues (3.13 has free-threading changes)
3. **fastembed internal batching**: the fastembed library may not batch properly for large sets, or its internal threadpool conflicts with synix's ThreadPoolExecutor
4. **Model download on first run**: bge-small-en-v1.5 needs to be downloaded on first use — could hang if network is slow or huggingface is unreachable

## Investigation Steps

1. Check if fastembed works standalone on salinas: `python -c "from fastembed import TextEmbedding; e = TextEmbedding(); list(e.embed(['test'] * 100))"`
2. Test with increasing batch sizes: 100, 1000, 5000, 10000, 23000 — find where it breaks
3. Check ONNX thread settings: `OMP_NUM_THREADS=1` or `ONNX_NUM_THREADS=1` as env vars
4. Check if synix's embedding code has a batch size parameter that can be tuned
5. Try a different embedding model or provider

## Current Workaround

Reference surface uses `modes=["fulltext"]` only. Main surface (small artifact count) uses `["fulltext", "semantic"]`.

## Impact

Semantic search on reference material (old docs, specs, intel, papers) is disabled. Fulltext search still works.
