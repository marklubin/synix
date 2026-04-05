#!/usr/bin/env bash
# Stdio-to-HTTP MCP proxy for the synix plugin.
# Reads JSON-RPC from stdin, POSTs to the synix server, writes responses to stdout.
# Server URL from plugin userConfig, env var, or default.

SERVER_URL="${CLAUDE_PLUGIN_OPTION_SERVER_URL:-${SYNIX_SERVER_URL:-http://localhost:8200}}"
MCP_URL="${SERVER_URL}/mcp"
SESSION_ID=""

while IFS= read -r line; do
    [ -z "$line" ] && continue

    HEADERS=(-H "Content-Type: application/json" -H "Accept: application/json")
    [ -n "$SESSION_ID" ] && HEADERS+=(-H "Mcp-Session-Id: $SESSION_ID")

    RESP=$(curl -sf --max-time 30 -D /dev/stderr "${HEADERS[@]}" -d "$line" "$MCP_URL" 2>/tmp/mcp_headers)

    # Capture session ID from response headers
    SID=$(grep -i 'mcp-session-id' /tmp/mcp_headers 2>/dev/null | tr -d '\r' | cut -d' ' -f2)
    [ -n "$SID" ] && SESSION_ID="$SID"

    if [ -n "$RESP" ]; then
        # Handle SSE responses (text/event-stream)
        if echo "$RESP" | head -1 | grep -q '^event:'; then
            echo "$RESP" | grep '^data: ' | sed 's/^data: //'
        else
            echo "$RESP"
        fi
    else
        # Server unreachable — return JSON-RPC error
        ID=$(echo "$line" | grep -o '"id":[^,}]*' | head -1 | cut -d: -f2)
        echo "{\"jsonrpc\":\"2.0\",\"id\":${ID:-null},\"error\":{\"code\":-32000,\"message\":\"Server unreachable at $MCP_URL\"}}"
    fi
done
