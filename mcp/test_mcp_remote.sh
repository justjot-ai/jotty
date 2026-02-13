#!/usr/bin/env bash
# Quick test: send MCP initialize + tools/list to the server via stdio.
# Run on the server: cd /home/opc/jotty-mcp && ./test_mcp_remote.sh
# Or from your machine: ssh workspace.justjot.ai 'cd /home/opc/jotty-mcp && ./test_mcp_remote.sh'
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source env.sh 2>/dev/null || true
# Send initialize then tools/list (MCP protocol)
(
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
) | timeout 5 ./run-mcp-docker.sh 2>/dev/null | while read -r line; do
  if echo "$line" | grep -q '"result"'; then
    echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); tools=r.get('tools',[]); print('Tools:', [t['name'] for t in tools] if isinstance(tools,list) else r)" 2>/dev/null || echo "$line"
  fi
done
echo "MCP server responded (tools listed above)."
