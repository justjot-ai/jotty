"""MCP server — exposes Jotty skills as Model Context Protocol tools.

Allows external MCP clients (Claude Desktop, Cursor, etc.) to use
Jotty's 164+ skills as tools.

Usage:
    python -m Jotty.core.capabilities.mcp_server

    # In Claude Desktop config (~/.claude/claude_desktop_config.json):
    # "mcpServers": {
    #     "jotty": {
    #         "command": "python",
    #         "args": ["-m", "Jotty.core.capabilities.mcp_server"]
    #     }
    # }
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

# Lazy registry singleton
_registry = None


def _get_registry():
    """Get or create the skills registry (singleton)."""
    global _registry
    if _registry is None:
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        _registry = get_skills_registry()
        if not _registry.initialized:
            _registry.init()
    return _registry


async def _run_tool(tool_fn, arguments: dict) -> Any:
    """Run a tool function, handling both sync and async callables."""
    if asyncio.iscoroutinefunction(tool_fn):
        return await tool_fn(arguments)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, tool_fn, arguments)


def _build_tool_list() -> List[Dict[str, Any]]:
    """Build the list of MCP tools from the skills registry."""
    registry = _get_registry()
    tools = []

    for skill_info in registry.list_skills():
        skill_name = skill_info["name"]
        skill = registry.get_skill(skill_name)
        if not skill:
            continue

        # Each skill's tool_metadata contains typed parameter schemas
        if skill._tool_metadata:
            for tool_name, meta in skill._tool_metadata.items():
                tools.append(
                    {
                        "name": f"{skill_name}__{tool_name}",
                        "description": meta.description or f"{skill_name}: {tool_name}",
                        "inputSchema": {
                            "type": "object",
                            "properties": meta.parameters.get("properties", {}),
                            "required": meta.parameters.get("required", []),
                        },
                    }
                )
        else:
            # Fallback: expose skill as single tool
            tools.append(
                {
                    "name": skill_name,
                    "description": skill.description or f"Jotty skill: {skill_name}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "params": {
                                "type": "object",
                                "description": "Parameters for the skill",
                            }
                        },
                    },
                }
            )

    return tools


async def _handle_call_tool(name: str, arguments: dict) -> str:
    """Handle a tool call from MCP client."""
    registry = _get_registry()

    # Parse compound name: "skill_name__tool_name"
    if "__" in name:
        skill_name, tool_name = name.split("__", 1)
        skill = registry.get_skill(skill_name)
        if not skill:
            return json.dumps({"error": f"Skill not found: {skill_name}"})
        tool_fn = skill.get_tool(tool_name)
        if not tool_fn:
            return json.dumps({"error": f"Tool not found: {tool_name} in {skill_name}"})
    else:
        # Single-name: use skill's first tool
        skill = registry.get_skill(name)
        if not skill:
            return json.dumps({"error": f"Skill not found: {name}"})
        tool_names = skill.list_tools()
        if not tool_names:
            return json.dumps({"error": f"Skill {name} has no tools"})
        tool_fn = skill.get_tool(tool_names[0])

    try:
        result = await _run_tool(tool_fn, arguments)
        if isinstance(result, dict):
            return json.dumps(result, default=str)
        return str(result)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return json.dumps({"error": str(e)})


def run_server() -> None:
    """Start the MCP stdio server.

    Requires the `mcp` package: pip install mcp
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        print("ERROR: MCP SDK not installed. Run: pip install mcp")
        print("See: https://github.com/modelcontextprotocol/python-sdk")
        raise SystemExit(1)

    app = Server("jotty")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        tool_defs = _build_tool_list()
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in tool_defs
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        result_text = await _handle_call_tool(name, arguments)
        return [TextContent(type="text", text=result_text)]

    async def _main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream)

    logger.info("Starting Jotty MCP server (stdio)")
    asyncio.run(_main())


if __name__ == "__main__":
    run_server()
