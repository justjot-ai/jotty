#!/usr/bin/env bash
# Run Jotty MCP HTTP/SSE server (endpoint with optional token).
# Usage:
#   ./run-mcp-http.sh
#   JOTTY_MCP_TOKEN=secret ./run-mcp-http.sh
#   MCP_HTTP_PORT=8767 JOTTY_GATEWAY_URL=http://justjot-processor:8766 ./run-mcp-http.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root = parent of Jotty (script is in Jotty/mcp/)
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec python3 -m Jotty.mcp.server_http
