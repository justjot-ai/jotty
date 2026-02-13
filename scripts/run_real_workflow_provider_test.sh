#!/usr/bin/env bash
# Run workflow provider tests against real n8n on pmi.workflows (SSH tunnel).
# Requires: N8N_API_KEY in env or in common/docker/.env.n8n-activepieces
#
# Usage:
#   cd /var/www/sites/personal/stock_market
#   export N8N_API_KEY='your-key-from-n8n-ui-settings-api'
#   Jotty/scripts/run_real_workflow_provider_test.sh
#
# Or with key from .env:
#   source common/docker/.env.n8n-activepieces 2>/dev/null
#   Jotty/scripts/run_real_workflow_provider_test.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Load N8N_API_KEY from common/docker/.env.n8n-activepieces if not set
if [ -z "${N8N_API_KEY:-}" ] && [ -f "common/docker/.env.n8n-activepieces" ]; then
  export $(grep -E '^N8N_API_KEY=' common/docker/.env.n8n-activepieces | xargs)
fi

echo "=== Real server test (pmi.workflows n8n) ==="
echo "  N8N_BASE_URL=http://localhost:5678 (tunnel)"
echo "  N8N_API_KEY=${N8N_API_KEY:-(not set)}"
echo ""

# Start tunnel in background
ssh -f -N -L 5678:127.0.0.1:5678 pmi.workflows || { echo "SSH tunnel failed. Is pmi.workflows in ~/.ssh/config?"; exit 1; }
sleep 2

cleanup() { pkill -f "ssh -f -N -L 5678:127.0.0.1:5678 pmi.workflows" 2>/dev/null; true; }
trap cleanup EXIT INT TERM

export N8N_BASE_URL=http://localhost:5678
python3 Jotty/scripts/test_workflow_providers_localhost.py
