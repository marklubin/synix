#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${CLAUDE_PLUGIN_OPTION_SERVER_URL:-${SYNIX_SERVER_URL:-http://salinas:8200}}"
CONTEXT=$(curl -sf --max-time 5 "${SERVER_URL}/api/v1/flat-file/context-doc" 2>/dev/null || echo "")

if [ -n "$CONTEXT" ]; then
    # Escape for JSON
    ESCAPED=$(echo "$CONTEXT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
    echo "{\"hookSpecificOutput\":{\"additionalContext\":${ESCAPED}}}"
fi
