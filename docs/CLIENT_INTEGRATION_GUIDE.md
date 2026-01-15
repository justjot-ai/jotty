# Jotty Client Integration Guide

## Overview

Jotty is designed as a **SaaS provider** that minimizes client code. This guide shows how to integrate Jotty with minimal effort.

## Architecture Philosophy

**Jotty (SaaS Provider)** handles:
- ✅ HTTP server setup
- ✅ Endpoint creation
- ✅ SSE formatting
- ✅ Authentication
- ✅ Error handling
- ✅ Logging
- ✅ Request validation
- ✅ Response formatting

**Client (JustJot.ai)** only needs to:
- ✅ Configure agents
- ✅ Run server
- ✅ That's it!

## Minimal Integration (3 Steps)

### Step 1: Install Jotty

```bash
pip install -e /path/to/Jotty
```

### Step 2: Create Server File

**File**: `supervisor/jotty_server.py` (50 lines)

```python
from Jotty.server import JottyServer, JottyServerConfig
from Jotty import AgentConfig, JottyConfig
import dspy

# Configure DSPy
dspy.configure(lm=dspy.LM('anthropic/claude-3-5-sonnet-20241022', api_key='...'))

# Define agents
agents = [
    AgentConfig(
        name="Research Assistant",
        agent=dspy.ChainOfThought("goal -> answer")
    )
]

# Create server
server = JottyServer(
    agents=agents,
    config=JottyConfig(),
    server_config=JottyServerConfig(
        port=8080,
        auth_type="clerk",  # or "bearer", "none"
        sse_format="usechat"
    )
)

# Run
server.run()
```

### Step 3: Run Server

```bash
python supervisor/jotty_server.py
```

**That's it!** You now have:
- ✅ `/api/chat/stream` - Chat streaming endpoint
- ✅ `/api/chat/execute` - Chat execution endpoint
- ✅ `/api/workflow/execute` - Workflow endpoint
- ✅ `/api/workflow/stream` - Workflow streaming
- ✅ `/api/agents` - Agent management
- ✅ `/api/health` - Health check

## What Jotty Provides

### 1. HTTP Server (`JottyHTTPServer`)
- Flask-based server with all endpoints
- CORS support
- Request/response handling
- Automatic error handling

### 2. SSE Formatting (`SSEFormatter`)
- `useChatFormatter` - Vercel AI SDK format
- `OpenAIFormatter` - OpenAI-compatible format
- `AnthropicFormatter` - Anthropic-compatible format
- `SSEFormatter` - Raw format

### 3. Authentication Middleware (`AuthMiddleware`)
- Bearer token authentication
- Clerk JWT support
- Custom validation functions
- Automatic token validation

### 4. Logging Middleware (`LoggingMiddleware`)
- Request/response logging
- Performance tracking
- Error logging

### 5. Error Handling (`ErrorMiddleware`)
- Graceful error responses
- Error logging
- Client-safe error messages

## Configuration Options

### Server Configuration

```python
JottyServerConfig(
    port=8080,                    # Server port
    host="0.0.0.0",              # Bind address
    debug=False,                  # Debug mode
    enable_cors=True,             # CORS support
    cors_origins=["*"],           # Allowed origins
    
    # Authentication
    auth_enabled=True,            # Enable auth
    auth_type="clerk",           # "bearer", "clerk", "none"
    auth_validate_fn=None,       # Custom validation
    
    # Logging
    enable_logging=True,          # Enable logging
    log_level="INFO",            # Log level
    
    # Error handling
    enable_error_handling=True,   # Enable error handling
    show_error_details=False,     # Show error details to clients
    
    # SSE Format
    sse_format="usechat",        # "usechat", "openai", "anthropic", "raw"
    
    # Health check
    health_check_path="/api/health"
)
```

## Integration Examples

### Example 1: Basic Chat Server

```python
from Jotty.server import JottyServer, JottyServerConfig
from Jotty import AgentConfig, JottyConfig
import dspy

dspy.configure(lm=dspy.LM('anthropic/claude-3-5-sonnet-20241022', api_key='...'))

server = JottyServer(
    agents=[AgentConfig(name="Assistant", agent=dspy.ChainOfThought("goal -> answer"))],
    server_config=JottyServerConfig(port=8080)
)
server.run()
```

### Example 2: With Custom Authentication

```python
def validate_clerk_token(token: str) -> bool:
    # Your Clerk token validation logic
    return len(token) > 20  # Example

server = JottyServer(
    agents=[...],
    server_config=JottyServerConfig(
        auth_type="clerk",
        auth_validate_fn=validate_clerk_token
    )
)
```

### Example 3: Multiple Agents

```python
agents = [
    AgentConfig(name="Research", agent=research_agent),
    AgentConfig(name="Code", agent=code_agent),
    AgentConfig(name="Analysis", agent=analysis_agent)
]

server = JottyServer(agents=agents)
server.run()
```

## Client SDK (TypeScript)

For TypeScript clients, use the Jotty client SDK:

```typescript
import { JottyClient } from '@jotty/client';

const client = new JottyClient('http://localhost:8080', {
  apiKey: 'your-api-key'
});

// Chat
const stream = await client.chat.stream('Hello', { history: [] });

// Workflow
const result = await client.workflow.execute('Analyze data');
```

## Migration from dspy_bridge_v2.py

**Before** (400+ lines):
- Manual Flask app setup
- Manual route creation
- Manual SSE formatting
- Manual error handling
- Manual authentication

**After** (50 lines):
- Import Jotty server
- Configure agents
- Run server

**Code Reduction**: ~87% less code!

## Benefits

1. **Minimal Code** - 50 lines vs 400+ lines
2. **Production Ready** - Built-in error handling, logging, auth
3. **Standardized** - Same API across all clients
4. **Maintainable** - Updates in Jotty benefit all clients
5. **Extensible** - Easy to add new capabilities

## Next Steps

1. Replace `dspy_bridge_v2.py` with `jotty_server.py`
2. Update frontend to use new endpoints (same API!)
3. Remove old bridge code
4. Enjoy minimal maintenance!
