"""
Jotty MCP Server - HTTP/SSE endpoint with optional token auth.

Exposes the same MCP tools as server.py (stdio) over HTTP so n8n or any client
can connect by URL instead of spawning a process.

- GET /sse          -> SSE stream (client subscribes)
- POST /messages/   -> JSON-RPC messages (with ?session_id= from first SSE event)

Auth: set JOTTY_MCP_TOKEN in env. Clients must send:
  Authorization: Bearer <token>
  or
  X-MCP-Token: <token>
If JOTTY_MCP_TOKEN is unset, no auth (use only on trusted networks).

Run: python -m Jotty.mcp.server_http
     JOTTY_MCP_TOKEN=secret python -m Jotty.mcp.server_http
Env: JOTTY_MCP_TOKEN, JOTTY_GATEWAY_URL, MCP_HTTP_HOST (default 0.0.0.0), MCP_HTTP_PORT (default 8767)
"""
import logging
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _file = Path(__file__).resolve()
    _root = _file.parent.parent.parent
    if _root.name == "stock_market" and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jotty-mcp-http")

# Token: if set, require it on every request
JOTTY_MCP_TOKEN = os.environ.get("JOTTY_MCP_TOKEN", "").strip()
MCP_HTTP_HOST = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
MCP_HTTP_PORT = int(os.environ.get("MCP_HTTP_PORT", "8767"))


def _get_token_from_scope(scope: dict) -> str | None:
    """Extract token from Authorization or X-MCP-Token in scope headers."""
    headers = scope.get("headers") or []
    auth = None
    x_token = None
    for (k, v) in headers:
        if k == b"authorization":
            auth = v.decode("latin1").strip()
            break
        if k.lower() == b"x-mcp-token":
            x_token = v.decode("latin1").strip()
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return x_token


def _unauthorized(scope, receive, send, msg: str = "Missing or invalid token"):
    from starlette.responses import Response
    response = Response(msg, status_code=401, headers={"WWW-Authenticate": "Bearer"})
    return response(scope, receive, send)


async def _check_token_and_dispatch(scope, receive, send, next_app):
    if scope["type"] != "http":
        await next_app(scope, receive, send)
        return
    if JOTTY_MCP_TOKEN:
        token = _get_token_from_scope(scope)
        if token != JOTTY_MCP_TOKEN:
            await _unauthorized(
                scope, receive, send,
                "Missing or invalid token (Authorization: Bearer <token> or X-MCP-Token: <token>)",
            )
            return
    await next_app(scope, receive, send)


# Import MCP app from stdio server
from Jotty.mcp.server import app as mcp_app

# SSE transport from MCP SDK
try:
    from mcp.server.sse import SseServerTransport
except ImportError as e:
    logger.error(
        "MCP SSE transport not available. Install: pip install 'mcp[server-sse]' starlette sse-starlette uvicorn"
    )
    raise SystemExit(1) from e

sse_transport = SseServerTransport("/messages/")


async def handle_sse_asgi(scope, receive, send):
    """GET /sse -> establish SSE; client then POSTs to /messages/?session_id=..."""
    try:
        async with sse_transport.connect_sse(scope, receive, send) as streams:
            await mcp_app.run(
                streams[0],
                streams[1],
                mcp_app.create_initialization_options(),
            )
    except ValueError as e:
        from starlette.responses import Response
        msg = str(e)
        status = 400 if "validation" in msg.lower() or "request" in msg.lower() else 500
        response = Response(msg, status_code=status)
        await response(scope, receive, send)
        return
    # SSE transport already sent the response; no second response


async def handle_post_messages_asgi(scope, receive, send):
    """POST /messages/?session_id=... -> JSON-RPC body."""
    await sse_transport.handle_post_message(scope, receive, send)


async def main_asgi(scope, receive, send):
    """Dispatch by path and method; token checked first."""
    if scope["type"] != "http":
        await send({"type": "http.disconnect"})
        return

    path = scope.get("path", "")
    method = scope.get("method", "")

    if path == "/sse" and method == "GET":
        await _check_token_and_dispatch(scope, receive, send, handle_sse_asgi)
        return
    if path.startswith("/messages") and method == "POST":
        await _check_token_and_dispatch(scope, receive, send, handle_post_messages_asgi)
        return

    from starlette.responses import Response
    response = Response(
        "Not Found. Use GET /sse and POST /messages/?session_id=...",
        status_code=404,
    )
    await response(scope, receive, send)


def main():
    try:
        import uvicorn
    except ImportError:
        logger.error("Install uvicorn: pip install uvicorn")
        raise SystemExit(1)
    logger.info(
        "Jotty MCP HTTP at http://%s:%s/sse (token auth=%s)",
        MCP_HTTP_HOST,
        MCP_HTTP_PORT,
        "on" if JOTTY_MCP_TOKEN else "off",
    )
    uvicorn.run(
        main_asgi,
        host=MCP_HTTP_HOST,
        port=MCP_HTTP_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
