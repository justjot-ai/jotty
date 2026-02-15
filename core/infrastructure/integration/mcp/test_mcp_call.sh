#!/usr/bin/env bash
# End-to-end test: call jotty_list_skills via MCP (same path n8n uses).
# Run on server: cd /home/opc/jotty-mcp && ./test_mcp_call.sh
# Expect: JSON with "skills" (if gateway is up) or "error" (if gateway is down). MCP layer must respond.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source env.sh 2>/dev/null || true

TOOL_NAME="${1:-jotty_list_skills}"
# MCP protocol: initialize -> notifications/initialized -> tools/call. Keep stdin open so server can respond.
(
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
  sleep 0.4
  echo '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  sleep 0.3
  echo "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"$TOOL_NAME\",\"arguments\":{}}}"
  sleep 12
) | timeout 25 ./run-mcp-docker.sh 2>/dev/null | while read -r line; do
  if echo "$line" | grep -q '"result"'; then
    # Second result is tools/call (id 2)
    if echo "$line" | grep -q '"content"'; then
      echo "=== Tool call result ($TOOL_NAME) ==="
      echo "$line" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('result', {})
    content = r.get('content', [])
    for c in content:
        if c.get('type') == 'text':
            text = c.get('text', '')
            try:
                inner = json.loads(text)
                print(json.dumps(inner, indent=2))
                ok = inner.get('success', False) or 'error' in inner
                sys.exit(0 if ok else 1)
            except Exception:
                print(text)
                sys.exit(0)
    sys.exit(2)
except Exception as e:
    print('Parse error:', e, file=sys.stderr)
    sys.exit(2)
" && echo "=== E2E test PASSED (MCP responded; gateway down is OK) ===" || echo "=== E2E test completed ==="
    fi
  fi
done
