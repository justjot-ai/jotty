# Jotty MCP as HTTP/SSE endpoint (with token)

Instead of n8n spawning the Jotty MCP server as a process (stdio), you can run the MCP server as an **HTTP/SSE endpoint** and point n8n (or any MCP client) at a URL with a **token**. No process spawn, no Python inside the n8n container.

## Why

- **n8n in Docker** doesn’t need to run the MCP server; it connects to a URL.
- **One MCP instance** can serve multiple clients (n8n, Cursor, etc.).
- **Token auth** keeps the endpoint safe when exposed (e.g. behind nginx or on an internal URL).

## Run the HTTP MCP server

### 1. Install dependencies

```bash
cd /var/www/sites/personal/stock_market
pip install -r Jotty/mcp/requirements.txt -r Jotty/mcp/requirements-http.txt
```

### 2. Start with token

```bash
export JOTTY_GATEWAY_URL="http://localhost:8766"   # or justjot-processor:8766 in Docker
export JOTTY_MCP_TOKEN="your-secret-token"

python -m Jotty.mcp.server_http
```

- Listens on `http://0.0.0.0:8767` by default.
- Override: `MCP_HTTP_HOST=127.0.0.1` `MCP_HTTP_PORT=8767`.

### 3. Endpoints

| Endpoint           | Method | Purpose |
|--------------------|--------|--------|
| `/sse`             | GET    | Open SSE stream; server sends first event with `endpoint` (path to POST messages). |
| `/messages/?session_id=<id>` | POST | Send JSON-RPC messages (session_id from the first SSE event). |

Clients must send the token on **every** request (GET /sse and POST /messages/):

- `Authorization: Bearer your-secret-token`
- or `X-MCP-Token: your-secret-token`

If `JOTTY_MCP_TOKEN` is **not** set, no auth is required (use only on trusted networks).

## n8n connection

If n8n supports **MCP over HTTP/SSE** (URL + headers):

1. **URL:** `http://<host>:8767/sse` (or your public/internal URL, e.g. `https://mcp.justjot.ai/sse` if you put the server behind nginx with TLS).
2. **Headers:**  
   `Authorization: Bearer your-secret-token`  
   or  
   `X-MCP-Token: your-secret-token`

If n8n only supports “command” (stdio) MCP, keep using the stdio server and the spawn approach; the HTTP endpoint is for clients that support URL-based MCP.

## Deploy on workspace (e.g. behind nginx)

1. Run the HTTP MCP server on the host or in a container (e.g. port 8767).
2. Optionally put nginx in front:  
   `https://mcp.justjot.ai/sse` → `http://127.0.0.1:8767/sse`  
   and forward `/messages/` to `http://127.0.0.1:8767/messages/`.
3. Clients use `https://mcp.justjot.ai/sse` with the token; nginx does not need to validate the token if the server is not publicly listed (or add a small auth snippet in nginx and keep `JOTTY_MCP_TOKEN` as a second layer).

## Test (E2E)

With the server running (no token for local test):

```bash
# Terminal 1
python -m Jotty.mcp.server_http

# Terminal 2
python Jotty/mcp/test_http_client.py
```

Expected: `Tools: ['jotty_chat', 'jotty_workflow', 'jotty_skill', 'jotty_list_skills']` and `OK`.

## Summary

| Item        | Value |
|------------|--------|
| Default URL | `http://0.0.0.0:8767/sse` |
| Token env  | `JOTTY_MCP_TOKEN` |
| Auth header | `Authorization: Bearer <token>` or `X-MCP-Token: <token>` |
| No token    | Auth disabled (trusted networks only) |
