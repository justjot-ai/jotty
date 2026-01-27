# JustJot.ai MCP over HTTP - Global Standard

## Overview

JustJot.ai MCP is now available as a **global standard** over HTTP, making it accessible from anywhere, not just local stdio connections.

## Architecture

### Transport Options

1. **stdio** (Local) - Used by Claude Desktop
   - Direct process communication
   - Fast, low latency
   - Requires local installation

2. **HTTP** (Remote) - Global standard ✨
   - RESTful API endpoints
   - Accessible from anywhere
   - Standard HTTP authentication
   - Perfect for Jotty skills and remote integrations

## API Endpoints

### Manifest
```
GET /api/mcp/manifest
```
Returns server capabilities and endpoint information.

### List Tools
```
GET /api/mcp/tools/list
```
Returns list of available MCP tools.

### Call Tool
```
POST /api/mcp/tools/call
Content-Type: application/json

{
  "name": "create_idea",
  "arguments": {
    "title": "My Idea",
    "description": "Description here"
  }
}
```

## Available Tools

- `list_ideas` - List all ideas
- `create_idea` - Create a new idea
- `get_idea` - Get a specific idea
- `update_idea` - Update an idea
- `delete_idea` - Delete an idea
- `get_ideas_by_tag` - Get ideas by tag

## Authentication

Uses Clerk authentication via `Authorization` header:
```
Authorization: Bearer <clerk_session_token>
```

## Usage in Jotty

### As a Skill

The `justjot-mcp-http` skill makes JustJot.ai MCP available in Jotty:

```python
from skills.justjot_mcp_http.tools import create_idea_tool

result = await create_idea_tool({
    'title': 'My Idea',
    'description': 'Description',
    'tags': ['tag1', 'tag2']
})
```

### Configuration

Set environment variables:
```bash
JUSTJOT_API_URL=https://justjot.ai
JUSTJOT_AUTH_TOKEN=<your_token>
```

## Benefits

1. **Global Access** - Use JustJot.ai MCP from anywhere
2. **Standard Protocol** - MCP-compatible HTTP transport
3. **Jotty Integration** - Available as skills in Jotty framework
4. **Discoverable** - Manifest endpoint for auto-discovery
5. **Scalable** - HTTP allows load balancing and caching

## Example: Remote MCP Client

```python
from skills.justjot_mcp_http.tools import get_client

client = get_client()
manifest = client.get_manifest()
tools = client.list_tools()
result = client.call_tool('create_idea', {'title': 'Test'})
```

## Making it a Global Standard

To make JustJot.ai MCP a global standard:

1. **Public Documentation** - Document the API endpoints
2. **Open Source** - Share the implementation
3. **MCP Registry** - Register in MCP server registry
4. **SDKs** - Provide SDKs for different languages
5. **Examples** - Provide integration examples

## Next Steps

1. ✅ HTTP endpoints implemented
2. ✅ Jotty skill created
3. ⏳ Add authentication middleware
4. ⏳ Add rate limiting
5. ⏳ Add streaming support (SSE)
6. ⏳ Create public documentation
7. ⏳ Register in MCP registry
