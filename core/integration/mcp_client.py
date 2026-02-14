"""
MCP Client for JustJot.ai

Provides programmatic access to JustJot.ai MCP tools via stdio transport.
Alternative to HTTP API approach - uses same protocol as Claude Desktop.
"""
import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP client that communicates with JustJot.ai MCP server via stdio.
    
    Uses same protocol as Claude Desktop - spawns subprocess and communicates
    via JSON-RPC over stdin/stdout.
    """
    
    def __init__(self, server_path: Optional[str] = None, env: Optional[Dict[str, str]] = None, mongodb_uri: Optional[str] = None) -> None:
        """
        Initialize MCP client.
        
        Args:
            server_path: Path to MCP server script (default: auto-detect)
            env: Environment variables for server process
            mongodb_uri: MongoDB connection string (default: from env or localhost)
        """
        self.server_path = server_path or self._find_server_path()
        
        # Prepare environment - include MongoDB URI
        self.env = env.copy() if env else {}
        
        # Set MongoDB URI (priority: parameter > env var > default)
        # IMPORTANT: JustJot.ai container on cmd.dev uses LOCAL MongoDB:
        # mongodb://justjot:ksG07jjmU9lO5zNd61W3Su9J@mongo:27017/justjot?authSource=admin
        # MCP client must use same database to match
        if mongodb_uri:
            self.env['MONGODB_URI'] = mongodb_uri
        elif 'MONGODB_URI' not in self.env:
            # Try to get from environment or use default
            # Default to local Docker MongoDB (same as JustJot.ai container uses)
            # From outside Docker, use localhost instead of 'mongo' hostname
            default_uri = os.getenv(
                'MONGODB_URI',
                'mongodb://justjot:ksG07jjmU9lO5zNd61W3Su9J@localhost:27017/justjot?authSource=admin'
            )
            # If URI points to planmyinvesting database, change to justjot
            # This ensures MCP client writes to same database as JustJot.ai API reads from
            if '/planmyinvesting' in default_uri:
                # Change database to justjot and add authSource if not present
                default_uri = default_uri.replace('/planmyinvesting', '/justjot')
                if 'authSource' not in default_uri:
                    default_uri += '?authSource=planmyinvesting'
            # If URI uses 'mongo' hostname (Docker internal), change to 'localhost' for external access
            if '@mongo:' in default_uri:
                default_uri = default_uri.replace('@mongo:', '@localhost:')
            self.env['MONGODB_URI'] = default_uri
        
        # Include Clerk secret if available
        if 'CLERK_SECRET_KEY' not in self.env:
            clerk_key = os.getenv('CLERK_SECRET_KEY')
            if clerk_key:
                self.env['CLERK_SECRET_KEY'] = clerk_key
        
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        
    def _find_server_path(self) -> str:
        """Find JustJot.ai MCP server path."""
        # Check common locations
        locations = [
            "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js",
            Path.home() / "JustJot.ai" / "dist" / "mcp" / "server.js",
            Path.cwd() / "JustJot.ai" / "dist" / "mcp" / "server.js"
        ]
        
        for loc in locations:
            if Path(loc).exists():
                return str(loc)
        
        raise FileNotFoundError(
            "JustJot.ai MCP server not found. "
            "Set server_path or ensure server.js exists."
        )
    
    async def connect(self) -> Any:
        """Start MCP server process and initialize connection."""
        if self.process:
            return  # Already connected
        
        # Prepare environment
        env = os.environ.copy()
        env.update(self.env)
        
        # Start server process
        self.process = subprocess.Popen(
            ["node", self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=0
        )
        
        # Start response reader
        asyncio.create_task(self._read_responses())
        
        # Initialize MCP connection
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "jotty-mcp-client",
                "version": "1.0.0"
            }
        })
        
        logger.info("MCP client connected")
    
    async def _read_responses(self) -> Any:
        """Read responses from MCP server."""
        if not self.process:
            return
        
        while True:
            line = await asyncio.to_thread(self.process.stdout.readline)
            if not line:
                break
            
            try:
                response = json.loads(line.strip())
                request_id = response.get("id")
                
                if request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    future.set_result(response)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from MCP server: {line}")
            except Exception as e:
                logger.error(f"Error reading MCP response: {e}")
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        if not self.process:
            await self.connect()
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        future = asyncio.Future()
        self.pending_requests[self.request_id] = future
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        # Wait for response
        response = await asyncio.wait_for(future, timeout=30.0)
        
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        return response.get("result", {})
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return result
    
    async def disconnect(self) -> Any:
        """Disconnect from MCP server."""
        if self.process:
            self.process.terminate()
            await asyncio.wait_for(
                asyncio.to_thread(self.process.wait),
                timeout=5.0
            )
            self.process = None
            logger.info("MCP client disconnected")
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        await self.disconnect()


# Convenience functions for common operations
async def call_justjot_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    server_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    mongodb_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call a JustJot.ai MCP tool using stdio transport.
    
    Args:
        tool_name: MCP tool name (e.g., "create_idea", "list_ideas")
        arguments: Tool arguments
        server_path: Path to MCP server (optional)
        env: Environment variables (optional)
        mongodb_uri: MongoDB connection string (optional)
    
    Returns:
        Tool execution result
    """
    async with MCPClient(server_path=server_path, env=env, mongodb_uri=mongodb_uri) as client:
        result = await client.call_tool(tool_name, arguments)
        # Parse result - MCP returns content array
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get('text', '')
                try:
                    return json.loads(text_content)
                except Exception:
                    return {'text': text_content}
        return result


import os
