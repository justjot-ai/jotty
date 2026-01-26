# MCP HTTP Endpoint Solution for cmd.dev

## Problem

MCP server uses **stdio transport** (stdin/stdout), which requires:
- Local subprocess spawning
- Direct MongoDB connection
- Can't work over HTTP/network

## Solution: Expose MCP via HTTP API

Since JustJot.ai is running on cmd.dev, we can:
1. **Use existing HTTP API endpoints** (`/api/ideas`, etc.)
2. **Create MCP-compatible HTTP endpoint** on cmd.dev
3. **Use internal API endpoints** (service-to-service)

## Option 1: Use Existing HTTP API (Current)

**Status**: ✅ Already works!

Our `mcp-justjot` skill uses HTTP API:
- Calls `/api/ideas` endpoints
- Works with cmd.dev deployment
- No MongoDB needed locally

**Just needs**: Correct cmd.dev URL

## Option 2: Create MCP HTTP Endpoint on cmd.dev

### Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐     Direct     ┌──────────┐
│  Jotty      │ ────────────> │  JustJot.ai      │ ────────────> │ MongoDB  │
│  (cmd.dev)  │   /mcp/tools │  MCP HTTP Server │   mongoose    │ (cmd.dev)│
└─────────────┘               └──────────────────┘               └──────────┘
```

### Implementation

**File**: `JustJot.ai/src/app/api/mcp/tools/route.ts`

```typescript
// HTTP endpoint that exposes MCP tools
export async function POST(request: Request) {
  const { name, arguments: args } = await request.json();
  
  // Connect to MongoDB (same as stdio server)
  await connectToDatabase();
  
  // Call tool handler (reuse from stdio server)
  const result = await handleToolCall(name, args);
  
  return Response.json({
    content: [{ type: 'text', text: result }],
    isError: false
  });
}
```

### Usage from Jotty

```python
from core.integration.mcp_client_http import MCPHTTPClient

# Use HTTP endpoint on cmd.dev
client = MCPHTTPClient(base_url="https://justjot.cmd.dev")
result = await client.call_tool("create_idea", {
    "title": "My Idea",
    "description": "Description"
})
```

## Option 3: Use Internal API Endpoints

**Status**: ✅ Already exists!

JustJot.ai has `/api/internal/*` endpoints for service-to-service calls:
- `/api/internal/ideas` - Create/list ideas
- `/api/internal/ideas/{id}` - Get/update idea
- No authentication required (internal service header)

**Our `mcp_tool_executor.py` already uses these!**

## Recommended Solution

**Use Option 1 (HTTP API) + Option 3 (Internal Endpoints)**

1. **Update `mcp-justjot` skill** to use `/api/internal/*` endpoints
2. **Set correct cmd.dev URL** in environment
3. **Use internal service header** (`x-internal-service: true`)

This way:
- ✅ Works with cmd.dev deployment
- ✅ No MongoDB needed locally
- ✅ Uses existing infrastructure
- ✅ No new endpoints needed

## Implementation

Update `skills/mcp-justjot/tools.py` to use internal endpoints:

```python
def _call_justjot_api(method, endpoint, ...):
    # Use /api/internal/* for service-to-service
    if endpoint.startswith('/api/ideas'):
        endpoint = endpoint.replace('/api/ideas', '/api/internal/ideas')
    
    headers = {
        "x-internal-service": "true",  # Bypass auth
        "Content-Type": "application/json"
    }
    # ... rest of code
```

## Answer to Your Question

**Q: How is Claude Code making it work if it's also running locally?**

**A**: Claude Code likely has:
- MongoDB running locally (or in Docker)
- OR MongoDB Atlas (remote MongoDB)
- OR MongoDB on same network

**Q: Can we get MCP endpoint on cmd.dev?**

**A**: Yes! Two options:
1. **Use existing HTTP API** (`/api/ideas`) - ✅ Already works
2. **Create `/api/mcp/tools` endpoint** - New HTTP endpoint for MCP protocol

**Recommendation**: Use existing HTTP API with `/api/internal/*` endpoints - no new code needed!
