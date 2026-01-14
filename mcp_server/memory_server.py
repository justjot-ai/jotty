#!/usr/bin/env python3
"""
MCP Server for Jotty Memory System
Exposes Jotty's 5-level memory as MCP tools for JustJot agents
"""
import sys
import json
import asyncio
from pathlib import Path

# Add Jotty and JustJot supervisor to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/var/www/sites/personal/stock_market/JustJot.ai/supervisor")

try:
    from mongodb_memory import MongoDBMemory
except ImportError:
    # Fallback: use mock memory for testing without MongoDB
    print("⚠️  MongoDB memory not available, using mock memory", file=sys.stderr)
    MongoDBMemory = None


class JottyMemoryMCPServer:
    """MCP server for Jotty memory operations"""

    def __init__(self, agent_name: str = "jotty-mcp"):
        self.agent_name = agent_name
        self.memory = None
        self.tools = [
            {
                "name": "jotty_store_memory",
                "description": "Store a memory in Jotty's hierarchical memory system (5 levels: EPISODIC, SEMANTIC, PROCEDURAL, META, CAUSAL)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content to store"
                        },
                        "level": {
                            "type": "string",
                            "enum": ["EPISODIC", "SEMANTIC", "PROCEDURAL", "META", "CAUSAL"],
                            "description": "Memory level (EPISODIC: events, SEMANTIC: facts, PROCEDURAL: how-to, META: learning strategies, CAUSAL: cause-effect)"
                        },
                        "context": {
                            "type": "object",
                            "description": "Metadata (tags, category, task_id, etc.)",
                            "properties": {
                                "category": {"type": "string"},
                                "task_id": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["content", "level"]
                }
            },
            {
                "name": "jotty_retrieve_memory",
                "description": "Retrieve relevant memories from Jotty's hierarchical memory using semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories"
                        },
                        "levels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["EPISODIC", "SEMANTIC", "PROCEDURAL", "META", "CAUSAL"]
                            },
                            "description": "Memory levels to search (default: all levels)"
                        },
                        "max_memories": {
                            "type": "number",
                            "description": "Maximum number of memories to return (default: 10)"
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "jotty_consolidate_memory",
                "description": "Trigger memory consolidation (episodic → semantic → procedural learning)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Agent to consolidate memories for (default: current agent)"
                        }
                    }
                }
            },
            {
                "name": "jotty_memory_stats",
                "description": "Get memory system statistics (count by level, recent memories, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Agent to get stats for (default: current agent)"
                        }
                    }
                }
            }
        ]

    def _initialize_memory(self):
        """Lazy initialization of memory system"""
        if self.memory is None:
            try:
                self.memory = MongoDBMemory(agent_name=self.agent_name)
                print(f"✅ Memory system initialized for agent: {self.agent_name}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  Warning: Could not connect to MongoDB: {e}", file=sys.stderr)
                print("   Memory operations will fail until MongoDB is available", file=sys.stderr)

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Handle MCP tool calls"""
        self._initialize_memory()

        if not self.memory:
            return {"error": "Memory system not initialized (MongoDB not available)"}

        try:
            if tool_name == "jotty_store_memory":
                memory_id = self.memory.store(
                    content=arguments["content"],
                    level=arguments["level"],
                    context=arguments.get("context", {})
                )
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "message": f"Stored {arguments['level']} memory"
                }

            elif tool_name == "jotty_retrieve_memory":
                memories = self.memory.retrieve(
                    query=arguments["query"],
                    levels=arguments.get("levels"),
                    budget_tokens=arguments.get("max_memories", 10) * 200,
                    category=arguments.get("category")
                )
                return {
                    "success": True,
                    "memories": memories,
                    "count": len(memories)
                }

            elif tool_name == "jotty_consolidate_memory":
                # Run consolidation
                agent_name = arguments.get("agent_name", self.agent_name)
                consolidation_result = self.memory.consolidate()
                return {
                    "success": True,
                    "result": consolidation_result,
                    "message": "Memory consolidation completed"
                }

            elif tool_name == "jotty_memory_stats":
                # Get memory statistics
                stats = self.memory.get_stats()
                return {
                    "success": True,
                    "stats": stats
                }

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    async def run_stdio(self):
        """Run MCP server using stdio protocol"""
        print("Jotty Memory MCP Server starting (stdio mode)...", file=sys.stderr)

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                method = request.get("method")

                if method == "tools/list":
                    # List available tools
                    response = {"tools": self.tools}

                elif method == "tools/call":
                    # Execute tool
                    params = request.get("params", {})
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})

                    result = await self.handle_tool_call(tool_name, arguments)
                    response = {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                else:
                    response = {"error": f"Unknown method: {method}"}

                # Send response
                print(json.dumps(response), flush=True)

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                response = {"error": str(e)}
                print(json.dumps(response), flush=True)

    async def run_http(self, port: int = 8082):
        """Run MCP server using HTTP (for easier testing)"""
        from aiohttp import web

        async def list_tools(request):
            return web.json_response({"tools": self.tools})

        async def call_tool(request):
            data = await request.json()
            tool_name = data.get("name")
            arguments = data.get("arguments", {})

            result = await self.handle_tool_call(tool_name, arguments)
            return web.json_response(result)

        app = web.Application()
        app.router.add_get('/tools/list', list_tools)
        app.router.add_post('/tools/call', call_tool)

        print(f"🚀 Jotty Memory MCP Server running on http://0.0.0.0:{port}", file=sys.stderr)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()

        # Keep running
        await asyncio.Event().wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jotty Memory MCP Server")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio",
                        help="Server mode (stdio for MCP protocol, http for testing)")
    parser.add_argument("--port", type=int, default=8082,
                        help="HTTP port (only used in http mode)")
    parser.add_argument("--agent", default="jotty-mcp",
                        help="Agent name for memory isolation")

    args = parser.parse_args()

    server = JottyMemoryMCPServer(agent_name=args.agent)

    if args.mode == "stdio":
        asyncio.run(server.run_stdio())
    else:
        asyncio.run(server.run_http(port=args.port))
