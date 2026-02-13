#!/usr/bin/env bash
# Wrapper to run Jotty MCP server inside Docker (Python 3.11). Use this as the
# "command" in n8n MCP config so n8n can spawn the server via stdio.
# Requires: Docker, image jotty-mcp (build with: docker build -t jotty-mcp .)
set -e
export JOTTY_GATEWAY_URL="${JOTTY_GATEWAY_URL:-http://host.docker.internal:8766}"
# Allow host to be reached from container (Linux: host.docker.internal may need extra_hosts)
exec docker run --rm -i \
  --add-host=host.docker.internal:host-gateway \
  -e JOTTY_GATEWAY_URL \
  jotty-mcp
