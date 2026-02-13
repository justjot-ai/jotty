"""
Jotty MCP Server - Expose Jotty as MCP tools for n8n, Cursor, Claude Desktop, etc.

Tools: jotty_chat, jotty_workflow, jotty_skill, jotty_list_skills

Uses Jotty gateway HTTP API (JOTTY_GATEWAY_URL, default http://localhost:8766).

Run (standalone):  python server.py     (from Jotty/mcp/ or deploy dir)
Run (as package):  python -m Jotty.mcp.server
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# When run as __main__ from package, ensure project root on path
if __name__ == "__main__":
    _file = Path(__file__).resolve()
    _root = _file.parent.parent.parent
    if _root.name == "stock_market" and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jotty-mcp")
# Uncomment for debug: logger.setLevel(logging.DEBUG)

# Gateway URL: env or default (same host)
JOTTY_GATEWAY_URL = os.environ.get("JOTTY_GATEWAY_URL", "http://localhost:8766").rstrip("/")


async def _post(path: str, body: dict) -> dict:
    """POST to Jotty gateway and return JSON."""
    import httpx
    url = f"{JOTTY_GATEWAY_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.exception("Gateway request failed: %s %s", path, e)
        return {"success": False, "error": str(e)}


async def _get(path: str) -> dict:
    """GET from Jotty gateway and return JSON."""
    import httpx
    url = f"{JOTTY_GATEWAY_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.exception("Gateway request failed: %s %s", path, e)
        return {"success": False, "error": str(e), "skills": []}


def _text_out(data: dict) -> list:
    """Return MCP TextContent list from a dict (e.g. API response)."""
    return [types.TextContent(type="text", text=json.dumps(data, indent=2))]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = Server("jotty")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="jotty_chat",
            description="Run Jotty chat (swarm/agents). Send a message and get a response. Use for open-ended tasks, research, or multi-step reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The task or question to send to Jotty"},
                    "session_id": {"type": "string", "description": "Optional session ID for conversation continuity"},
                    "history": {"type": "array", "description": "Optional list of prior messages for context"},
                },
                "required": ["message"],
            },
        ),
        types.Tool(
            name="jotty_workflow",
            description="Run Jotty workflow mode (goal-oriented execution). Use for multi-step goals like 'research and summarize' or 'find and email'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The goal to achieve"},
                    "session_id": {"type": "string", "description": "Optional session ID"},
                },
                "required": ["goal"],
            },
        ),
        types.Tool(
            name="jotty_skill",
            description="Run a single Jotty skill by name. Use for specific actions (e.g. web_search, send_telegram). Call jotty_list_skills first to see names.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of the skill (e.g. web-search)"},
                    "params": {"type": "object", "description": "Parameters for the skill", "default": {}},
                },
                "required": ["skill_name"],
            },
        ),
        types.Tool(
            name="jotty_list_skills",
            description="List available Jotty skills (names and descriptions). Use before calling jotty_skill.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    args = arguments or {}
    if name == "jotty_chat":
        message = args.get("message", "")
        if not message:
            return _text_out({"success": False, "error": "message is required"})
        body = {"message": message}
        if "session_id" in args:
            body["session_id"] = args["session_id"]
        if "history" in args:
            body["history"] = args["history"]
        out = await _post("/api/chat", body)
        return _text_out(out)
    if name == "jotty_workflow":
        goal = args.get("goal", "")
        if not goal:
            return _text_out({"success": False, "error": "goal is required"})
        body = {"goal": goal}
        if "session_id" in args:
            body["session_id"] = args["session_id"]
        out = await _post("/api/workflow", body)
        return _text_out(out)
    if name == "jotty_skill":
        skill_name = args.get("skill_name", "")
        if not skill_name:
            return _text_out({"success": False, "error": "skill_name is required"})
        params = args.get("params", {})
        out = await _post(f"/api/skill/{skill_name}", params)
        return _text_out(out)
    if name == "jotty_list_skills":
        out = await _get("/api/skills")
        return _text_out(out)
    return _text_out({"success": False, "error": f"Unknown tool: {name}"})


async def main_async() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main() -> None:
    try:
        import anyio
        # Prefer trio if available (mcp stdio uses anyio); else asyncio
        try:
            anyio.run(main_async, backend="trio")
        except LookupError:
            anyio.run(main_async, backend="asyncio")
    except ImportError:
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
