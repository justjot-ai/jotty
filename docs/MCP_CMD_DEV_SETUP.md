# MCP Client Setup for cmd.dev

## Summary

✅ **MCP Client works on cmd.dev!**

- ✅ MCP server is compiled and available
- ✅ Node.js is available (v22.21.0)
- ✅ MCP client connects successfully
- ✅ Can list tools (25 tools found)
- ⚠️  Need correct MongoDB URI for idea creation

## Current Status

### What Works

1. **MCP Client Connection**: ✅
   - Connects to MCP server via stdio
   - Lists 25 available tools
   - Communication protocol working

2. **MCP Server**: ✅
   - Compiled at: `/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js`
   - Node.js available: v22.21.0

### What Needs Configuration

1. **MongoDB URI**: ⚠️
   - Default (`mongodb://localhost:27017/justjot`) not accessible
   - Need to set correct MongoDB URI from environment

## Setup Instructions

### 1. Set MongoDB URI

**Option A: From JustJot.ai .env.local**

```bash
export MONGODB_URI="mongodb://planmyinvesting:aRpOVx2HYl6jS9LO@planmyinvesting.com:27017/justjot"
```

**Option B: Use Remote MongoDB**

```bash
export MONGODB_URI="mongodb://user:password@host:27017/justjot"
```

**Option C: Use Local MongoDB (if running)**

```bash
export MONGODB_URI="mongodb://localhost:27017/justjot"
```

### 2. Set Clerk Secret Key (Optional)

```bash
export CLERK_SECRET_KEY="your_clerk_secret_key"
```

### 3. Test MCP Client

```bash
cd /var/www/sites/personal/stock_market/Jotty
python3 test_mcp_cmd_dev.py
```

## Test Results

### Successful Connection

```
🧪 Testing MCP Client on cmd.dev...
   MongoDB URI: mongodb://localhost:27017/justjot
   MCP Server: /var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js
   Server exists: True

🔌 Connecting to MCP server...
✅ Connected!

📋 Listing tools...
✅ Found 25 tools
   Tools: list_ideas, get_idea, create_idea, update_idea, delete_idea...
```

### MongoDB Connection Error

```
❌ Failed: connect ECONNREFUSED 127.0.0.1:27017
```

**Solution**: Set correct `MONGODB_URI` environment variable.

## Comparison: MCP Client vs HTTP API

### MCP Client (Direct MongoDB)

**Pros**:
- ✅ Faster (direct DB access)
- ✅ Same as Claude Desktop
- ✅ Better for testing MCP functionality
- ✅ No HTTP overhead

**Cons**:
- ❌ Requires MongoDB access
- ❌ Requires correct MongoDB URI
- ❌ Requires Node.js and compiled MCP server

### HTTP API

**Pros**:
- ✅ Works from any server
- ✅ No MongoDB access needed
- ✅ Uses existing API infrastructure

**Cons**:
- ❌ HTTP overhead
- ❌ Requires API server running

## Recommendation

**For cmd.dev**: Use **MCP Client** approach

**Why**:
1. cmd.dev has MongoDB accessible (with correct URI)
2. Faster and more direct
3. Tests actual MCP client functionality
4. Same as Claude Desktop (better comparison)

**Setup**:
1. Set `MONGODB_URI` environment variable
2. Use `mcp-justjot-mcp-client` skill
3. Test with `test_mcp_cmd_dev.py`

## Next Steps

1. **Find correct MongoDB URI**:
   ```bash
   # Check JustJot.ai .env.local
   grep MONGODB_URI JustJot.ai/.env.local
   ```

2. **Test with correct URI**:
   ```bash
   export MONGODB_URI="correct_uri_here"
   python3 test_mcp_cmd_dev.py
   ```

3. **Use in skills**:
   - Use `mcp-justjot-mcp-client` skill
   - It will use MCP client automatically
   - Falls back to HTTP API if MongoDB not accessible
