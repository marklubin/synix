#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SYNIX_SERVER_URL:-http://salinas:8200}"

# Read hook input from stdin
INPUT=$(cat)

# Extract session ID from the hook input JSON
SESSION_ID=$(echo "$INPUT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("sessionId",""))' 2>/dev/null || echo "")

if [ -z "$SESSION_ID" ]; then
    exit 0
fi

# Find the session file
# Claude Code stores sessions in ~/.claude/projects/{encoded-path}/{sessionId}.jsonl
SESSION_FILE=""
for dir in ~/.claude/projects/*/; do
    candidate="${dir}${SESSION_ID}.jsonl"
    if [ -f "$candidate" ]; then
        SESSION_FILE="$candidate"
        break
    fi
done

if [ -z "$SESSION_FILE" ] || [ ! -f "$SESSION_FILE" ]; then
    exit 0
fi

# Read and push the session content
FILENAME="${SESSION_ID}.jsonl"

# POST to synix server
python3 -c "
import json, urllib.request

session_file = '${SESSION_FILE}'
filename = '${FILENAME}'
server_url = '${SERVER_URL}'

data = json.dumps({'content': open(session_file).read(), 'filename': filename}).encode()
req = urllib.request.Request(
    server_url + '/api/v1/ingest/sessions',
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    urllib.request.urlopen(req, timeout=10)
except Exception:
    pass  # silently fail — never block session end
" 2>/dev/null || true
