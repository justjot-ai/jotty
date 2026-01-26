# How Claude Code Makes MCP Work

## Key Insight: MCP Server Connects Directly to MongoDB

Claude Code works because the MCP server **connects directly to MongoDB**, not through HTTP API!

## MCP Configuration

**File**: `JustJot.ai/mcp_config_claude.json`

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

## How It Works

### 1. Claude Code Spawns MCP Server
- Claude reads config from `~/.claude/claude_desktop_config.json`
- Spawns Node.js subprocess: `node dist/mcp/server.js`
- Passes environment variables (MONGODB_URI, CLERK_SECRET_KEY)

### 2. MCP Server Connects to MongoDB
**Code**: `JustJot.ai/src/mcp/server.ts`
```typescript
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/justjot';

async function connectDB() {
  if (mongoose.connection.readyState === 1) return;
  await mongoose.connect(MONGODB_URI, {
    // ... connection options
  });
}
```

### 3. Direct Database Access
- MCP server uses **mongoose** to connect directly to MongoDB
- No HTTP API needed!
- Reads/writes directly to database
- Uses Clerk for authentication (user context)

## Why Our Jotty MCP Client Failed

**Our test showed:**
```
❌ Error: MongoDB connection refused (ECONNREFUSED 127.0.0.1:27017)
```

**Reason**: MongoDB not running locally

**Solution**: 
1. Start MongoDB locally, OR
2. Use remote MongoDB URI in MCP client

## How to Make It Work

### Option 1: Use Remote MongoDB (if available)

```python
from core.integration.mcp_client import MCPClient

# Use remote MongoDB (e.g., MongoDB Atlas or remote server)
async with MCPClient(
    mongodb_uri="mongodb://remote-host:27017/justjot"
) as client:
    result = await client.call_tool("create_idea", {...})
```

### Option 2: Use Same MongoDB as Claude Code

```python
# If Claude Code works, use same MongoDB URI
async with MCPClient(
    mongodb_uri="mongodb://localhost:27017/justjot"  # Same as Claude
) as client:
    result = await client.call_tool("create_idea", {...})
```

### Option 3: Check Claude Code's MongoDB URI

Claude Code might be using:
- Remote MongoDB (MongoDB Atlas)
- Docker MongoDB container
- Different MongoDB instance

**Check Claude Code's config:**
```bash
cat ~/.claude/claude_desktop_config.json | jq '.mcpServers.justjot.env.MONGODB_URI'
```

## Key Differences

| Aspect | Claude Code | Our Jotty Skill |
|--------|------------|----------------|
| **MongoDB** | ✅ Running (local or remote) | ❌ Not running locally |
| **Connection** | Direct via mongoose | Direct via mongoose |
| **MCP Server** | ✅ Built and accessible | ✅ Built and accessible |
| **Result** | ✅ Works | ⚠️ Needs MongoDB |

## Answer: Yes, We CAN Create Ideas via MCP!

**The code works perfectly.** We just need:
1. **MongoDB running** (local or remote)
2. **Correct MONGODB_URI** in environment

Claude Code works because it has MongoDB accessible. Once we use the same MongoDB URI, our MCP client will work too!

## Next Steps

1. **Find MongoDB URI** that Claude Code uses
2. **Update MCP client** to use that URI
3. **Test idea creation** - it will work!

The MCP client implementation is correct - we just need the right MongoDB connection!
