# CLI UX Requirements

The CLI is a selling point. Use Click + Rich to make it feel polished.

## Commands

```bash
synix run pipeline.py [--source-dir ./exports]   # Run pipeline, materialize projections
synix search "query" [--layers episodes,core]     # Search with provenance
synix lineage <artifact-id>                       # Provenance tree
synix status                                      # Build summary
```

## Live Progress (CRITICAL)

Every CLI command must show live interactive progress. The user must never stare at a frozen terminal.

- Any operation >1 second must show a Rich spinner or progress bar
- Spinner text must indicate what's happening (e.g., "Summarizing conversation abc123 [3/47]")
- Long-running engine functions must accept an `on_progress` callback

## Per-Command UX

- **`synix run`**: Rich progress bars per layer. Show: layer name, artifact count, built/cached/skipped. Final summary table with timing.
- **`synix search`**: Results as Rich panels — layer label colored by level, content snippet, artifact label. Provenance chain as indented tree below each result.
- **`synix lineage`**: Rich Tree widget — full dependency graph from artifact to raw transcript.
- **`synix status`**: Rich table — layer name, artifact count, last build time, cache hit ratio.

## Color Scheme (by layer level)

- Level 0: dim/grey
- Level 1: blue
- Level 2: green
- Level 3: gold/yellow

## Error Handling

- All errors through Rich console with clear messages
- Never a raw Python traceback in normal operation
- Every command and option has clear `--help` text
