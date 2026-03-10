# Chunked Search

Split documents into paragraph-level chunks and search across them — no LLM required.

## What This Demonstrates

- **Chunk transform (1:N)** — each source document becomes multiple chunk artifacts
- **No LLM calls** — `Chunk` is pure text processing, no API key needed
- **Provenance tracking** — every chunk traces back to its source document
- **Chunk metadata** — `source_label`, `chunk_index`, `chunk_total` for downstream grouping
- **Search over chunks** — full-text search across all chunk artifacts

## Pipeline

```
articles [source]  →  chunks [Chunk, separator="\n\n"]  →  search [SynixSearch]
```

## Run It

```bash
uvx synix build pipeline.py
uvx synix release HEAD --to local
uvx synix search 'encryption' --release local
```

## Try It

- Search for `"CRDT merge"` to find architecture details
- Search for `"pricing"` to find the product overview chunks
- Run `uvx synix lineage chunks-t-text-architecture-guide-2` to see provenance
- Rebuild with `uvx synix build pipeline.py` — everything is cached (instant)
