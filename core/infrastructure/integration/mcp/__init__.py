"""
Jotty MCP Server - Expose Jotty chat, workflow, and skills as MCP tools.

Run: python -m Jotty.mcp.server
Or:  python Jotty/mcp/server.py

Requires Jotty gateway running (e.g. http://localhost:8766) or set JOTTY_GATEWAY_URL.
"""

from .server import main  # type: ignore[attr-defined]

__all__ = ["main"]
