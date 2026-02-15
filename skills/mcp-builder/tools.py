"""
MCP Builder Skill - Create and validate MCP servers.

Helps build MCP (Model Context Protocol) servers with templates,
validation, and best practices.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("mcp-builder")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def create_mcp_server_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new MCP server project.

    Args:
        params:
            - server_name (str): Name of server
            - language (str, optional): 'python' or 'node'
            - output_directory (str, optional): Output directory
            - include_examples (bool, optional): Include examples

    Returns:
        Dictionary with created files and paths
    """
    status.set_callback(params.pop("_status_callback", None))

    server_name = params.get("server_name", "")
    language = params.get("language", "python")
    output_directory = params.get("output_directory", ".")
    include_examples = params.get("include_examples", True)

    if not server_name:
        return {"success": False, "error": "server_name is required"}

    if language not in ["python", "node"]:
        return {"success": False, "error": 'language must be "python" or "node"'}

    try:
        output_path = Path(os.path.expanduser(output_directory))
        server_path = output_path / server_name
        server_path.mkdir(parents=True, exist_ok=True)

        files_created = []

        if language == "python":
            # Create Python MCP server
            files_created.extend(
                await _create_python_server(server_path, server_name, include_examples)
            )
        else:
            # Create Node.js MCP server
            files_created.extend(
                await _create_node_server(server_path, server_name, include_examples)
            )

        return {"success": True, "server_path": str(server_path), "files_created": files_created}

    except Exception as e:
        logger.error(f"MCP server creation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _create_python_server(
    server_path: Path, server_name: str, include_examples: bool
) -> List[str]:
    """Create Python MCP server structure."""

    files_created = []

    # Create main server file
    server_py = server_path / "server.py"
    server_content = f'''"""
{server_name} MCP Server

Model Context Protocol server implementation.
"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("{server_name}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="example_tool",
            description="Example tool",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "input": {{"type": "string", "description": "Input parameter"}}
                }},
                "required": ["input"]
            }}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "example_tool":
        input_value = arguments.get("input", "")
        return [TextContent(
            type="text",
            text=f"Example tool called with: {{input_value}}"
        )]

    raise ValueError(f"Unknown tool: {{name}}")

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''

    server_py.write_text(server_content, encoding="utf-8")
    files_created.append("server.py")

    # Create requirements.txt
    requirements = server_path / "requirements.txt"
    requirements.write_text("mcp>=0.1.0\n", encoding="utf-8")
    files_created.append("requirements.txt")

    # Create README
    readme = server_path / "README.md"
    readme_content = f"""# {server_name}

MCP Server implementation.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

## Tools

- `example_tool`: Example tool description
"""

    readme.write_text(readme_content, encoding="utf-8")
    files_created.append("README.md")

    return files_created


async def _create_node_server(
    server_path: Path, server_name: str, include_examples: bool
) -> List[str]:
    """Create Node.js MCP server structure."""

    files_created = []

    # Create package.json
    package_json = server_path / "package.json"
    package_content = """{
  "name": "mcp-server",
  "version": "1.0.0",
  "description": "MCP Server",
  "main": "server.js",
  "type": "module",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.1.0"
  }
}
"""

    package_json.write_text(package_content, encoding="utf-8")
    files_created.append("package.json")

    # Create server.js
    server_js = server_path / "server.js"
    server_content = """import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

from Jotty.core.infrastructure.utils.skill_status import SkillStatus

const server = new Server({
  name: "mcp-server",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
  },
});

server.setRequestHandler("tools/list", async () => ({
  tools: [
    {
      name: "example_tool",
      description: "Example tool",
      inputSchema: {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "Input parameter",
          },
        },
        required: ["input"],
      },
    },
  ],
}));

server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "example_tool") {
    const input = request.params.arguments?.input || "";
    return {
      content: [
        {
          type: "text",
          text: `Example tool called with: ${input}`,
        },
      ],
    };
  }

  throw new Error(`Unknown tool: ${request.params.name}`);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP server running on stdio");
}

main().catch(console.error);
"""

    server_js.write_text(server_content, encoding="utf-8")
    files_created.append("server.js")

    # Create README
    readme = server_path / "README.md"
    readme_content = f"""# {server_name}

MCP Server implementation.

## Setup

```bash
npm install
```

## Run

```bash
npm start
```

## Tools

- `example_tool`: Example tool description
"""

    readme.write_text(readme_content, encoding="utf-8")
    files_created.append("README.md")

    return files_created


@async_tool_wrapper()
async def validate_mcp_server_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an MCP server structure.

    Args:
        params:
            - server_path (str): Path to server directory

    Returns:
        Dictionary with validation results
    """
    status.set_callback(params.pop("_status_callback", None))

    server_path = params.get("server_path", "")

    if not server_path:
        return {"success": False, "error": "server_path is required"}

    server_dir = Path(os.path.expanduser(server_path))

    if not server_dir.exists():
        return {"success": False, "error": f"Server directory not found: {server_path}"}

    issues = []
    warnings = []

    # Check for main server file
    python_server = server_dir / "server.py"
    node_server = server_dir / "server.js"

    if not python_server.exists() and not node_server.exists():
        issues.append("No server file found (server.py or server.js)")

    # Check for package files
    if python_server.exists():
        requirements = server_dir / "requirements.txt"
        if not requirements.exists():
            warnings.append("requirements.txt not found (recommended for Python)")

    if node_server.exists():
        package_json = server_dir / "package.json"
        if not package_json.exists():
            issues.append("package.json is required for Node.js servers")

    # Check for README
    readme = server_dir / "README.md"
    if not readme.exists():
        warnings.append("README.md not found (recommended)")

    valid = len(issues) == 0

    return {"success": True, "valid": valid, "issues": issues, "warnings": warnings}
