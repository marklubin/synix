# Project Goals

## Q2 2026 Objectives

1. **Knowledge Server v1** — Event-driven build queue, local GPU inference, prompt management
2. **Docker Deployment** — Containerized server with GPU passthrough, volume mounts for data persistence
3. **Viewer Overhaul** — Pipeline DAG visualization, build status, prompt editor tabs
4. **Quality** — Fix [Object object] metadata bug, group chunks by source document

## Key Metrics
- Build throughput: 79+ tok/s on RTX 3060 with Qwen3.5-2B
- Full rebuild of 500 source documents in under 30 minutes
- Zero cloud API dependency for inference
