# JustJot.ai MCP HTTP - Complete Guide

## Overview

JustJot.ai MCP is now available as a **global standard** over HTTP, making it accessible from anywhere without requiring local installation or browser-based OAuth.

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
   - **SSE streaming support** for real-time responses

## API Endpoints

### 1. Manifest (Discovery)
```
GET /api/mcp/manifest
```
Returns server capabilities and endpoint information.

**Response:**
```json
{
  "name": "justjot-ai",
  "version": "1.0.0",
  "transport": {
    "type": "http",
    "endpoints": {
      "tools": {
        "list": "https://justjot.ai/api/mcp/tools/list",
        "call": "https://justjot.ai/api/mcp/tools/call"
      }
    }
  }
}
```

### 2. List Tools
```
GET /api/mcp/tools/list
```
Returns list of available MCP tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "create_idea",
      "description": "Create a new idea",
      "inputSchema": { ... }
    }
  ]
}
```

### 3. Call Tool (Standard)
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

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"success\":true,\"id\":\"...\",\"message\":\"Idea created\"}"
    }
  ],
  "isError": false
}
```

### 4. Call Tool (Streaming)
```
POST /api/mcp/tools/call/stream
Content-Type: application/json
Accept: text/event-stream

{
  "name": "create_idea",
  "arguments": {
    "title": "My Idea"
  }
}
```

**Response (SSE Stream):**
```
data: {"type":"start","tool":"create_idea"}

data: {"type":"progress","message":"Processing..."}

data: {"type":"result","content":[{"type":"text","text":"..."}],"isError":false}

data: {"type":"complete"}
```

### 5. API Documentation
```
GET /api/mcp/docs
```
Returns complete API documentation in JSON format.

## Authentication

### Method 1: API Key + User ID (Recommended for Jotty)

**Headers:**
```
x-api-key: sk_test_... (CLERK_SECRET_KEY)
x-user-id: user_36W0zSjAkJe54fkRtDWLb4qMrpH
```

**Use Case:** Service-to-service, automated systems, Jotty skills

**Configuration:**
```bash
JUSTJOT_API_KEY=sk_test_...
JUSTJOT_USER_ID=user_36W0zSjAkJe54fkRtDWLb4qMrpH
```

### Method 2: Bearer Token

**Header:**
```
Authorization: Bearer <clerk_session_token>
```

**Use Case:** Browser-based integrations, user-initiated actions

### Method 3: Clerk Session (Browser)

**Header:**
```
Cookie: <clerk_session_cookie>
```

**Use Case:** Standard browser requests

## Rate Limiting

### Limits

- **Per User:** 60 requests/minute
- **Per API Key:** 100 requests/minute
- **Global:** 1000 requests/minute

### Rate Limit Headers

When rate limit is exceeded (429 status):

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1738000000000
Retry-After: 45
```

### Handling Rate Limits

```python
result = client.call_tool('create_idea', {'title': 'Test'})

if result.get('error') == 'Rate limit exceeded':
    retry_after = result.get('retry_after', 60)
    print(f"Rate limited. Retry after {retry_after} seconds")
```

## Available Tools

### list_ideas
List all ideas for the authenticated user.

**Parameters:**
- `status` (optional): Filter by status
- `limit` (optional): Maximum results (default: 50)
- `tags` (optional): Filter by tags (array)

### create_idea
Create a new idea.

**Parameters:**
- `title` (required): Idea title
- `description` (optional): Description
- `tags` (optional): Tags (array)
- `status` (optional): Status (default: Draft)
- `sections` (optional): Initial sections
- `userId` (optional): User ID (defaults to authenticated user)
- `author` (optional): Author name

### get_idea
Get a specific idea by ID.

**Parameters:**
- `id` (required): Idea ID

### update_idea
Update an existing idea.

**Parameters:**
- `id` (required): Idea ID
- `title` (optional): Updated title
- `description` (optional): Updated description
- `status` (optional): Updated status
- `tags` (optional): Updated tags
- `sections` (optional): Updated sections

### delete_idea
Delete an idea.

**Parameters:**
- `id` (required): Idea ID

### get_ideas_by_tag
Get ideas filtered by tag.

**Parameters:**
- `tag` (required): Tag to filter by
- `limit` (optional): Maximum results (default: 50)

## Usage in Jotty

### Basic Usage

```python
from skills.justjot_mcp_http.tools import create_idea_tool

result = await create_idea_tool({
    'title': 'My Idea',
    'description': 'Created via Jotty',
    'tags': ['jotty', 'automated']
})

if result.get('success'):
    idea_id = result.get('idea_id')
    print(f"Created idea: {idea_id}")
```

### Using the Client Directly

```python
from skills.justjot_mcp_http.tools import get_client

client = get_client()

# List tools
tools = client.list_tools()

# Call tool
result = client.call_tool('create_idea', {
    'title': 'Test',
    'tags': ['test']
})

# Get manifest
manifest = client.get_manifest()
```

### Streaming (SSE)

```python
from skills.justjot_mcp_http.tools import get_client

client = get_client()

# Stream tool execution
for event in client.call_tool('create_idea', {'title': 'Test'}, stream=True):
    if event.get('type') == 'start':
        print("Started...")
    elif event.get('type') == 'progress':
        print(f"Progress: {event.get('message')}")
    elif event.get('type') == 'result':
        print(f"Result: {event.get('content')}")
    elif event.get('type') == 'complete':
        print("Done!")
    elif event.get('type') == 'error':
        print(f"Error: {event.get('error')}")
```

## Error Handling

### Authentication Errors (401)

```python
result = client.call_tool('create_idea', {'title': 'Test'})

if not result.get('success'):
    if 'Authentication required' in result.get('error', ''):
        print("Check your API key and user ID")
```

### Rate Limit Errors (429)

```python
result = client.call_tool('create_idea', {'title': 'Test'})

if result.get('error') == 'Rate limit exceeded':
    retry_after = result.get('retry_after', 60)
    print(f"Rate limited. Wait {retry_after} seconds")
```

### Tool Errors

```python
result = client.call_tool('create_idea', {'title': ''})

if not result.get('success'):
    error = result.get('error')
    print(f"Tool error: {error}")
```

## Configuration

### Environment Variables

```bash
# Jotty/.env
JUSTJOT_API_URL=https://justjot.ai
JUSTJOT_API_KEY=sk_test_...
JUSTJOT_USER_ID=user_36W0zSjAkJe54fkRtDWLb4qMrpH

# Alternative: Use Bearer token
# JUSTJOT_AUTH_TOKEN=<session_token>
```

### Programmatic Configuration

```python
from skills.justjot_mcp_http.tools import JustJotMCPHTTPClient

client = JustJotMCPHTTPClient(
    base_url='https://justjot.ai',
    api_key='sk_test_...',
    user_id='user_xxx'
)
```

## Making it a Global Standard

### 1. Public API
- ✅ Deployed to `https://justjot.ai/api/mcp/*`
- ✅ Public documentation endpoint
- ✅ CORS enabled for discovery

### 2. MCP Registry
- Register in MCP server registry
- Add to MCP discovery
- Versioning support

### 3. SDKs
- ✅ Python SDK (Jotty skill)
- ⏳ JavaScript/TypeScript SDK
- ⏳ Other languages

### 4. Examples
- ✅ Usage examples in docs
- ✅ Integration examples
- ⏳ Tutorial videos

## Best Practices

1. **Use API Key Method for Automation**
   - Store keys securely
   - Never commit to git
   - Use environment variables

2. **Handle Rate Limits**
   - Implement exponential backoff
   - Cache responses when possible
   - Monitor rate limit headers

3. **Use Streaming for Long Operations**
   - Better UX for users
   - Real-time progress updates
   - Handle errors gracefully

4. **Error Handling**
   - Always check `success` field
   - Handle rate limits
   - Log errors for debugging

## Testing

### Test Authentication

```bash
curl -X POST https://justjot.ai/api/mcp/tools/call \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_..." \
  -H "x-user-id: user_xxx" \
  -d '{"name":"list_ideas","arguments":{"limit":5}}'
```

### Test Streaming

```bash
curl -X POST https://justjot.ai/api/mcp/tools/call/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_..." \
  -H "x-user-id: user_xxx" \
  -d '{"name":"create_idea","arguments":{"title":"Test"}}'
```

## Next Steps

1. ✅ HTTP endpoints implemented
2. ✅ Authentication (API key + session)
3. ✅ Rate limiting
4. ✅ SSE streaming
5. ✅ Public documentation
6. ⏳ Register in MCP registry
7. ⏳ Create JavaScript SDK
8. ⏳ Add more tools (templates, sections, etc.)
