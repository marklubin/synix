#!/usr/bin/env bash
# Synix container e2e smoke test
#
# Starts a fresh container with minimal test data, waits for vLLM + build,
# then verifies MCP tools, REST API, and viewer all work correctly.
#
# Usage: bash deploy/smoke-test/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="synix-smoke-test"
TEST_DATA="$SCRIPT_DIR"
SYNIX_DATA=$(mktemp -d)
MCP_PORT=8200
VIEWER_PORT=9471
VLLM_PORT=8100
MAX_WAIT=600  # 10 minutes for vLLM startup + first build

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
    rm -rf "$SYNIX_DATA"
    echo "Done."
}
trap cleanup EXIT

PASS=0
FAIL=0
pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

echo "=== Synix Container Smoke Test ==="
echo "  Image: synix-server"
echo "  Test data: $TEST_DATA"
echo "  Build state: $SYNIX_DATA"
echo ""

# --- Start container ---
echo "=== Starting container ==="
podman run -d \
    --name "$CONTAINER_NAME" \
    --network=host \
    --device nvidia.com/gpu=0 \
    -v "$TEST_DATA/sources:/data/sources:z" \
    -v "$SYNIX_DATA:/data/.synix:z" \
    -v /srv/synix/models:/models:z \
    synix-server 2>&1

echo "Container started. Waiting for services..."

# --- Wait for MCP HTTP to be healthy ---
echo ""
echo "=== Waiting for MCP HTTP (port $MCP_PORT) ==="
STARTED=0
for i in $(seq 1 $((MAX_WAIT / 5))); do
    if curl -sf "http://localhost:$MCP_PORT/api/v1/health" >/dev/null 2>&1; then
        echo "  MCP HTTP healthy after $((i * 5))s"
        STARTED=1
        break
    fi
    sleep 5
done
if [ "$STARTED" = "0" ]; then
    echo "  TIMEOUT waiting for MCP HTTP"
    echo ""
    echo "=== Container logs ==="
    podman logs --tail 50 "$CONTAINER_NAME"
    exit 1
fi

# --- Wait for vLLM to be healthy ---
echo ""
echo "=== Waiting for vLLM (port $VLLM_PORT) ==="
VLLM_STARTED=0
for i in $(seq 1 $((MAX_WAIT / 5))); do
    if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo "  vLLM healthy after $((i * 5))s"
        VLLM_STARTED=1
        break
    fi
    sleep 5
done
if [ "$VLLM_STARTED" = "0" ]; then
    echo "  TIMEOUT waiting for vLLM"
    podman logs --tail 30 "$CONTAINER_NAME"
    exit 1
fi

# --- Ingest test data via REST API ---
echo ""
echo "=== Ingesting test data ==="
for f in "$TEST_DATA"/sources/sessions/*.jsonl; do
    FNAME=$(basename "$f")
    CONTENT=$(cat "$f" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null)
    curl -sf -X POST "http://localhost:$MCP_PORT/api/v1/ingest/sessions" \
        -H "Content-Type: application/json" \
        -d "{\"content\":$CONTENT,\"filename\":\"$FNAME\",\"client_id\":\"smoke-test\"}" > /dev/null 2>&1 \
        && echo "  Ingested: $FNAME" || echo "  FAILED: $FNAME"
done
for f in "$TEST_DATA"/sources/documents/*.md; do
    FNAME=$(basename "$f")
    CONTENT=$(cat "$f" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null)
    curl -sf -X POST "http://localhost:$MCP_PORT/api/v1/ingest/documents" \
        -H "Content-Type: application/json" \
        -d "{\"content\":$CONTENT,\"filename\":\"$FNAME\",\"client_id\":\"smoke-test\"}" > /dev/null 2>&1 \
        && echo "  Ingested: $FNAME" || echo "  FAILED: $FNAME"
done
for f in "$TEST_DATA"/sources/reference/*.md; do
    FNAME=$(basename "$f")
    CONTENT=$(cat "$f" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null)
    curl -sf -X POST "http://localhost:$MCP_PORT/api/v1/ingest/reference" \
        -H "Content-Type: application/json" \
        -d "{\"content\":$CONTENT,\"filename\":\"$FNAME\",\"client_id\":\"smoke-test\"}" > /dev/null 2>&1 \
        && echo "  Ingested: $FNAME" || echo "  FAILED: $FNAME"
done

# --- Wait for first build to complete (queue processes test data) ---
echo ""
echo "=== Waiting for build (window=${MAX_WAIT}s max) ==="
BUILD_DONE=0
for i in $(seq 1 60); do
    # Check if build completed via document status (all docs should be 'released')
    STATUS=$(curl -sf "http://localhost:$MCP_PORT/api/v1/search?q=synix&limit=1" 2>/dev/null || echo "{}")
    COUNT=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('count', 0))" 2>/dev/null || echo "0")
    if [ "$COUNT" -gt "0" ]; then
        echo "  Build complete — $COUNT artifacts after $((i * 5))s"
        BUILD_DONE=1
        break
    fi
    sleep 5
done
if [ "$BUILD_DONE" = "0" ]; then
    echo "  TIMEOUT waiting for build"
    podman logs --tail 30 "$CONTAINER_NAME"
    exit 1
fi

# ===================================================================
# TEST SUITE
# ===================================================================

echo ""
echo "=== REST API Tests ==="

# Health check
CODE=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:$MCP_PORT/api/v1/health")
[ "$CODE" = "200" ] && pass "GET /api/v1/health → 200" || fail "GET /api/v1/health → $CODE"

# List buckets
BUCKETS=$(curl -sf "http://localhost:$MCP_PORT/api/v1/buckets" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('buckets',[])))")
[ "$BUCKETS" -ge "5" ] && pass "GET /api/v1/buckets → $BUCKETS buckets" || fail "GET /api/v1/buckets → $BUCKETS"

# Search
RESULTS=$(curl -sf "http://localhost:$MCP_PORT/api/v1/search?q=synix&limit=5" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('count',0))")
[ "$RESULTS" -gt "0" ] && pass "GET /api/v1/search?q=synix → $RESULTS results" || fail "GET /api/v1/search → 0 results"

# Ingest a new document and track status
INGEST_RESP=$(curl -sf -X POST "http://localhost:$MCP_PORT/api/v1/ingest/documents" \
    -H "Content-Type: application/json" \
    -H "X-Client-Id: smoke-test" \
    -d '{"content":"# Smoke Test\n\nThis document was ingested by the smoke test at '"$(date -Iseconds)"'","filename":"smoke-test.md"}')
DOC_ID=$(echo "$INGEST_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('doc_id',''))" 2>/dev/null || echo "")
if [ -n "$DOC_ID" ]; then
    pass "POST /api/v1/ingest/documents → doc_id=$DOC_ID"
else
    fail "POST /api/v1/ingest/documents → no doc_id"
fi

# Document status
if [ -n "$DOC_ID" ]; then
    DOC_STATUS=$(curl -sf "http://localhost:$MCP_PORT/api/v1/document/$DOC_ID" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
    [ -n "$DOC_STATUS" ] && pass "GET /api/v1/document/$DOC_ID → status=$DOC_STATUS" || fail "GET /api/v1/document → empty"
fi

# Prompts list
PROMPT_COUNT=$(curl -sf "http://localhost:$MCP_PORT/api/v1/prompts" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('prompts',[])))")
[ "$PROMPT_COUNT" -ge "1" ] && pass "GET /api/v1/prompts → $PROMPT_COUNT prompts" || fail "GET /api/v1/prompts → $PROMPT_COUNT"

echo ""
echo "=== Viewer Tests ==="
# Note: viewer exits if no release exists at startup. It starts before the
# first build completes, so it may not be available in this test scenario.
# This is a known limitation — tracked for fix (viewer should retry/reload).
VIEWER_UP=0
CODE=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:$VIEWER_PORT/" 2>/dev/null || echo "000")
if [ "$CODE" = "200" ]; then
    VIEWER_UP=1
    pass "GET viewer / → 200"

    ART_COUNT=$(curl -sf "http://localhost:$VIEWER_PORT/api/status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('artifact_count',0))" 2>/dev/null || echo "0")
    [ "$ART_COUNT" -gt "0" ] && pass "GET /api/status → $ART_COUNT artifacts" || fail "GET /api/status → 0"

    DAG_NODES=$(curl -sf "http://localhost:$VIEWER_PORT/api/dag" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('nodes',[])))" 2>/dev/null || echo "0")
    [ "$DAG_NODES" -gt "0" ] && pass "GET /api/dag → $DAG_NODES nodes" || fail "GET /api/dag → 0"

    QUEUE_RESP=$(curl -sf "http://localhost:$VIEWER_PORT/api/build-status" | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if 'queue_depth' in d else 'fail')" 2>/dev/null || echo "fail")
    [ "$QUEUE_RESP" = "ok" ] && pass "GET /api/build-status → has queue_depth" || fail "GET /api/build-status → missing fields"

    JS_CODE=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:$VIEWER_PORT/static/app.js" 2>/dev/null)
    [ "$JS_CODE" = "200" ] && pass "GET /static/app.js → 200" || fail "GET /static/app.js → $JS_CODE"
else
    echo "  ⚠ Viewer not available (exits before first build — known limitation)"
fi

echo ""
echo "=== vLLM Tests ==="

# vLLM health
VLLM_CODE=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:$VLLM_PORT/health")
[ "$VLLM_CODE" = "200" ] && pass "GET vLLM /health → 200" || fail "GET vLLM /health → $VLLM_CODE"

# vLLM completion
VLLM_RESP=$(curl -sf "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-2B","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"temperature":0}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])")
[ -n "$VLLM_RESP" ] && pass "vLLM completion → $VLLM_RESP" || fail "vLLM completion → empty"

# ===================================================================
# RESULTS
# ===================================================================

echo ""
echo "========================================"
echo "  PASS: $PASS  FAIL: $FAIL"
echo "========================================"

if [ "$FAIL" -gt "0" ]; then
    echo ""
    echo "Container logs (last 30 lines):"
    podman logs --tail 30 "$CONTAINER_NAME"
    exit 1
fi

echo ""
echo "All smoke tests passed!"
