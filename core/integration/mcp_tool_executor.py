"""
MCP Tool Executor for DSPy
Enables DSPy signatures to call MCP tools from JustJot
"""
import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str  # Which MCP server provides this tool


class MCPToolExecutor:
    """Execute MCP tools from DSPy agents"""

    def __init__(self, mcp_config_path: str = None, base_url: str = None):
        """
        Initialize with MCP server configuration

        Args:
            mcp_config_path: Path to mcp.json or claude_desktop_config.json
            base_url: Base URL for JustJot API (default: JUSTJOT_API_URL env var)
        """
        self.mcp_config_path = mcp_config_path or self._find_mcp_config()
        # Use environment variable or cmd.dev or Docker service name as fallback
        if base_url:
            self.base_url = base_url
        else:
            # Check environment variables in priority order
            url = (
                os.getenv("JUSTJOT_API_URL") or 
                os.getenv("NEXT_PUBLIC_API_URL") or 
                os.getenv("JUSTJOT_BASE_URL")
            )
            
            if url:
                self.base_url = url
            else:
                # Fallback to cmd.dev (production) or Docker service
                self.base_url = "https://justjot.ai.cmd.dev"  # cmd.dev deployment
        self.available_tools: List[MCPTool] = []
        self.tool_map: Dict[str, MCPTool] = {}

    def _find_mcp_config(self) -> str:
        """Find MCP configuration file"""
        # Check common locations
        locations = [
            Path.home() / ".claude" / "claude_desktop_config.json",
            Path.cwd() / "mcp.json",
            Path("/var/www/sites/personal/stock_market/JustJot.ai/mcp.json"),
        ]

        for loc in locations:
            if loc.exists():
                return str(loc)

        # Return default path (will create if needed)
        return str(Path.cwd() / "mcp.json")

    async def discover_tools(self) -> List[MCPTool]:
        """
        Discover available MCP tools from configured servers

        Returns:
            List of available MCP tools
        """
        # Currently hardcoded for JustJot MCP tools
        # NOTE: Dynamic MCP protocol discovery would require implementing
        # the full MCP discovery protocol. For now, explicitly defining tools
        # provides better reliability and type safety.

        self.available_tools = [
            # Idea operations
            MCPTool(
                name="mcp__justjot__get_idea",
                description="Get a single idea by ID with all its sections and content",
                input_schema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The MongoDB ObjectId of the idea"}
                    },
                    "required": ["id"]
                },
                server="justjot"
            ),
            MCPTool(
                name="mcp__justjot__list_ideas",
                description="List all ideas with optional filtering by status, template, or tags",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["Draft", "Published", "Archived"],
                            "description": "Filter by status"
                        },
                        "tag": {"type": "string", "description": "Filter by tag"},
                        "limit": {"type": "number", "description": "Maximum number of ideas to return (default: 20)"}
                    }
                },
                server="justjot"
            ),
            MCPTool(
                name="mcp__justjot__search_ideas",
                description="Search ideas by title, description, or content. Supports full-text search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "number", "description": "Maximum results (default: 10)"}
                    },
                    "required": ["query"]
                },
                server="justjot"
            ),
            MCPTool(
                name="mcp__justjot__create_idea",
                description="Create a new idea with optional sections",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title of the idea"},
                        "description": {"type": "string", "description": "Brief description of the idea"},
                        "sections": {
                            "type": "array",
                            "description": "Initial sections for the idea",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "type": {"type": "string", "default": "text"}
                                }
                            }
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for the idea"
                        }
                    },
                    "required": ["title"]
                },
                server="justjot"
            ),
            # Section operations
            MCPTool(
                name="mcp__justjot__add_section",
                description="Add a new section to an existing idea",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ideaId": {"type": "string", "description": "The MongoDB ObjectId of the idea"},
                        "title": {"type": "string", "description": "Section title"},
                        "content": {"type": "string", "description": "Section content", "default": ""},
                        "type": {"type": "string", "description": "Section type (text, code, etc.)", "default": "text"}
                    },
                    "required": ["ideaId", "title"]
                },
                server="justjot"
            ),
        ]

        self.tool_map = {tool.name: tool for tool in self.available_tools}

        return self.available_tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments (must match input_schema)

        Returns:
            Tool execution result
        """
        tool = self.tool_map.get(tool_name)
        if not tool:
            raise ValueError(
                f"Tool {tool_name} not found. Available: {list(self.tool_map.keys())}"
            )

        # Validate arguments against schema (basic validation)
        required = tool.input_schema.get("required", [])
        for field in required:
            if field not in arguments:
                raise ValueError(f"Missing required argument: {field}")

        # Call MCP server
        result = await self._call_mcp_server(tool.server, tool_name, arguments)

        return result

    async def _call_mcp_server(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call MCP server via HTTP API"""
        if server_name == "justjot":
            return await self._call_justjot_mcp(tool_name, arguments)

        raise NotImplementedError(f"MCP server {server_name} not implemented")

    async def _call_justjot_mcp(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call JustJot MCP tools via HTTP API"""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for MCP tool execution. Install: pip install aiohttp")

        # Map MCP tool to API endpoint
        # Use /api/internal/* endpoints for service-to-service calls (no auth required)
        endpoint_map = {
            "mcp__justjot__get_idea": ("GET", "/api/internal/ideas/{id}"),
            "mcp__justjot__list_ideas": ("GET", "/api/internal/ideas"),
            "mcp__justjot__search_ideas": ("POST", "/api/internal/ideas/search"),
            "mcp__justjot__create_idea": ("POST", "/api/internal/ideas"),
            "mcp__justjot__add_section": ("POST", "/api/internal/ideas/{ideaId}/sections"),
        }

        endpoint_info = endpoint_map.get(tool_name)
        if not endpoint_info:
            raise ValueError(f"Unknown JustJot tool: {tool_name}")

        method, endpoint = endpoint_info

        # Format endpoint with path parameters
        if "{id}" in endpoint and "id" in arguments:
            endpoint = endpoint.format(id=arguments["id"])
            arguments = {k: v for k, v in arguments.items() if k != "id"}
        elif "{ideaId}" in endpoint and "ideaId" in arguments:
            endpoint = endpoint.format(ideaId=arguments["ideaId"])
            arguments = {k: v for k, v in arguments.items() if k != "ideaId"}

        url = f"{self.base_url}{endpoint}"
        
        # Add internal service header for service-to-service calls
        headers = {
            "x-internal-service": "true",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                if method == "GET":
                    async with session.get(url, params=arguments, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                else:  # POST
                    async with session.post(url, json=arguments, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as e:
                # If blue fails, try green (for Docker blue-green deployment)
                if "justjot-ai-blue" in self.base_url:
                    fallback_url = url.replace("justjot-ai-blue", "justjot-ai-green")
                    try:
                        if method == "GET":
                            async with session.get(fallback_url, params=arguments, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                                resp.raise_for_status()
                                return await resp.json()
                        else:  # POST
                            async with session.post(fallback_url, json=arguments, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                                resp.raise_for_status()
                                return await resp.json()
                    except Exception as fallback_err:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Blue-green fallback also failed: {fallback_err}"
                        )
                raise RuntimeError(f"Failed to call JustJot API {url}: {e}")

    def format_tools_for_dspy(self) -> str:
        """
        Format available tools for DSPy signature

        Returns:
            String describing available tools for LLM
        """
        if not self.available_tools:
            return "No tools available"

        tool_descriptions = []
        for tool in self.available_tools:
            # Format schema nicely
            props = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])

            params = []
            for name, schema in props.items():
                param_str = f"{name}: {schema.get('type', 'any')}"
                if name in required:
                    param_str += " (required)"
                if "description" in schema:
                    param_str += f" - {schema['description']}"
                params.append(param_str)

            tool_descriptions.append(
                f"**{tool.name}**\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n  " + "\n  ".join(params)
            )

        return "\n\n".join(tool_descriptions)

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tool_map.keys())
