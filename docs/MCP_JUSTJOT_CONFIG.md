# MCP JustJot.ai Configuration Guide

## API URL Configuration

JustJot.ai can run in different environments. The MCP skill automatically detects the correct URL using this priority order:

### Priority Order:
1. `JUSTJOT_API_URL` environment variable (highest priority)
2. `NEXT_PUBLIC_API_URL` environment variable
3. `JUSTJOT_BASE_URL` environment variable
4. `https://justjot.ai.cmd.dev` (cmd.dev deployment)
5. `http://localhost:3000` (local development fallback)

### Environment Variables

**For cmd.dev deployment:**
```bash
export JUSTJOT_API_URL="https://justjot.ai.cmd.dev"
```

**For Docker deployment:**
```bash
export JUSTJOT_API_URL="http://justjot-ai-blue:3000"
```

**For local development:**
```bash
export JUSTJOT_API_URL="http://localhost:3000"
```

### Authentication

If JustJot.ai requires authentication, set:
```bash
export JUSTJOT_AUTH_TOKEN="your-auth-token"
```

The skill will automatically include this in API requests as:
```
Authorization: Bearer your-auth-token
```

## Usage Examples

### cmd.dev Deployment
```python
import os
os.environ['JUSTJOT_API_URL'] = 'https://justjot.ai.cmd.dev'

from skills.mcp_justjot.tools import list_ideas_tool
ideas = await list_ideas_tool({})
```

### Docker Deployment
```python
import os
os.environ['JUSTJOT_API_URL'] = 'http://justjot-ai-blue:3000'

from skills.mcp_justjot.tools import create_idea_tool
idea = await create_idea_tool({
    'title': 'My Idea',
    'description': 'Description'
})
```

### With Authentication
```python
import os
os.environ['JUSTJOT_API_URL'] = 'https://justjot.ai.cmd.dev'
os.environ['JUSTJOT_AUTH_TOKEN'] = 'your-token'

from skills.mcp_justjot.tools import list_ideas_tool
ideas = await list_ideas_tool({})
```

## Testing Connection

```python
from skills.mcp_justjot.tools import list_ideas_tool

result = await list_ideas_tool({})
if result.get('success'):
    print(f"✅ Connected! Found {result.get('count', 0)} ideas")
else:
    print(f"❌ Connection failed: {result.get('error')}")
```

## Troubleshooting

### Connection Refused
- Check if JustJot.ai is running
- Verify the API URL is correct
- Check network connectivity (for cmd.dev)

### Authentication Errors
- Verify `JUSTJOT_AUTH_TOKEN` is set correctly
- Check if token has expired
- Ensure token has required permissions

### CORS Issues
- If calling from browser, ensure CORS is configured on JustJot.ai
- Use server-side calls when possible
