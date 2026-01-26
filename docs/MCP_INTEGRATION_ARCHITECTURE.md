# MCP Integration Architecture

## How Claude Accesses JustJot.ai MCP

### Claude Desktop Configuration

Claude Desktop connects to MCP servers via configuration file:

**Location:** `~/.claude/claude_desktop_config.json` or `~/.claude/mcp.json`

**Configuration:**
```json
{
  "mcpServers": {
    "justjot": {
      "command": "node",
      "args": ["/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"],
      "env": {
        "MONGODB_URI": "mongodb://localhost:27017/justjot",
        "CLERK_SECRET_KEY": "${CLERK_SECRET_KEY}"
      }
    }
  }
}
```

### Communication Protocol

1. **Stdio Transport**: Claude Desktop spawns the MCP server as a subprocess
2. **JSON-RPC**: Communication via stdin/stdout using JSON-RPC protocol
3. **Tool Discovery**: Claude calls `tools/list` to discover available tools
4. **Tool Execution**: Claude calls `tools/call` with tool name and arguments

### MCP Server Implementation

**TypeScript Server** (`src/mcp/server.ts`):
- Uses `@modelcontextprotocol/sdk`
- Exposes tools: `list_ideas`, `create_idea`, `update_idea`, etc.
- Connects to MongoDB directly
- Uses Clerk for authentication

**Python Server** (`mcp_server.py`):
- Uses `mcp` Python package (FastMCP)
- Same tools as TypeScript version
- Can be used as alternative

## Jotty Skill Integration Options

### Option A: HTTP API (Current Implementation) ✅

**Approach**: Call JustJot.ai REST API endpoints directly

**Pros:**
- Simple HTTP requests
- No subprocess management
- Works from any environment
- Can use cmd.dev URL

**Cons:**
- Requires API to be running
- Needs authentication handling
- Not using MCP protocol directly

**Implementation**: `skills/mcp-justjot/tools.py`
- Uses `requests` library
- Calls `/api/ideas`, `/api/templates`, etc.
- Configurable via `JUSTJOT_API_URL`

### Option B: MCP Protocol Client (Future Enhancement)

**Approach**: Use MCP client library to connect to MCP server

**Pros:**
- Uses same protocol as Claude Desktop
- Consistent with MCP standard
- Can reuse MCP server

**Cons:**
- Requires subprocess management
- More complex (stdio communication)
- Needs MCP client library

**Implementation** (Potential):
```python
from mcp import ClientSession, StdioServerParameters
import asyncio

async def call_mcp_tool(tool_name: str, arguments: dict):
    server_params = StdioServerParameters(
        command="node",
        args=["/path/to/mcp/server.js"]
    )
    
    async with ClientSession(server_params) as session:
        result = await session.call_tool(tool_name, arguments)
        return result
```

### Option C: Hybrid Approach (Recommended)

**Approach**: Use HTTP API for Jotty skills, MCP protocol for Claude Desktop

**Why:**
- Jotty skills need HTTP for cmd.dev deployment
- Claude Desktop uses stdio for local subprocess
- Both access same JustJot.ai backend
- Best of both worlds

## Current Implementation

### Jotty Skill: `mcp-justjot`

**Location**: `skills/mcp-justjot/tools.py`

**How it works:**
1. Uses HTTP API calls to JustJot.ai
2. Configurable URL (defaults to cmd.dev)
3. Supports authentication via `JUSTJOT_AUTH_TOKEN`
4. Wraps MCP tools as Jotty skills

**Tools Available:**
- `list_ideas_tool`
- `create_idea_tool`
- `get_idea_tool`
- `update_idea_tool`
- `delete_idea_tool`
- `list_templates_tool`
- `get_template_tool`
- `add_section_tool`
- `update_section_tool`
- `list_tags_tool`

### MCP Server: `src/mcp/server.ts`

**How Claude Desktop uses it:**
1. Claude reads config from `~/.claude/claude_desktop_config.json`
2. Spawns Node.js subprocess with server.js
3. Communicates via stdio using JSON-RPC
4. Server connects to MongoDB directly
5. Uses Clerk for user authentication

## Comparison

| Aspect | Claude Desktop (MCP) | Jotty Skill (HTTP) |
|--------|---------------------|-------------------|
| Protocol | JSON-RPC over stdio | HTTP REST API |
| Transport | Subprocess stdin/stdout | HTTP requests |
| URL | Local subprocess | cmd.dev or localhost |
| Auth | Clerk secret key | Bearer token |
| Use Case | Claude Desktop integration | Jotty workflows |
| Deployment | Local machine | Any environment |

## Recommendation

**Keep both approaches:**

1. **MCP Server** (`src/mcp/server.ts`): For Claude Desktop integration
   - Uses stdio transport
   - Direct MongoDB access
   - Clerk authentication

2. **Jotty Skill** (`skills/mcp-justjot`): For Jotty workflows
   - Uses HTTP API
   - Works with cmd.dev deployment
   - Can be used in composite skills/pipelines

Both access the same JustJot.ai backend, just via different protocols.
