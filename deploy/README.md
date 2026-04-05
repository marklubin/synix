# Synix Server — Container Deployment

GPU-accelerated knowledge server running vLLM (Qwen3.5-2B) + synix serve in a single container.

## Prerequisites

- NVIDIA GPU with ≥12GB VRAM
- Podman (or Docker) with NVIDIA Container Toolkit
- CDI configured: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`

## Host Directory Layout

```
/srv/synix/
├── project/
│   └── sources/          ← your data (sessions, documents, reference, etc.)
├── data/                 ← .synix build state (created automatically)
└── models/               ← HuggingFace model cache (created automatically)
```

## Quick Start

```bash
# Create host directories
sudo mkdir -p /srv/synix/{data,models}
sudo chown $USER:$USER /srv/synix/{data,models}

# Build and run
cd deploy/
podman-compose up -d

# Watch logs (first run downloads model, ~5GB)
podman-compose logs -f

# Check health
curl http://localhost:8200/health
```

## Services (host networking)

| Service | Port | Description |
|---------|------|-------------|
| MCP HTTP + REST API | 8200 | Ingest, search, document status, prompt CRUD |
| Viewer | 9471 | Browse, Pipeline DAG, Prompts editor |
| vLLM | 8100 | Local inference (internal, managed by synix serve) |

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `/srv/synix/project/sources` | `/data/sources` | Source data buckets (sessions, docs, reference) |
| `/srv/synix/data` | `/data/.synix` | Build state (objects, refs, queue, prompts) |
| `/srv/synix/models` | `/models` | HuggingFace model cache |

## Configuration

Edit `synix-server.toml` to change:
- `[vllm]` — model, GPU device, context length, memory utilization
- `[auto_build]` — build queue window (seconds)
- `[buckets.*]` — source bucket paths and patterns

Edit `pipeline.py` to change:
- Source definitions, transform layers, search surfaces, projections

Rebuild container after changing either file:
```bash
podman-compose down && podman-compose up -d --build
```

## GPU Selection

To use a different GPU, change in `compose.yml`:
```yaml
devices:
  - nvidia.com/gpu=1    # use GPU 1 instead of 0
```

And in `synix-server.toml`:
```toml
[vllm]
gpu_device = 1
```

## Updating Synix

```bash
cd /path/to/synix
git pull
cd deploy/
podman-compose down && podman-compose up -d --build
```
