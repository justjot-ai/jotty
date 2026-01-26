# Running MCP Client on cmd.dev

## Overview

Since cmd.dev has both MongoDB and JustJot.ai API running, we can use the **MCP client approach** (direct MongoDB) instead of HTTP API. This is better for testing because:

- ✅ **Faster** - Direct MongoDB connection (no HTTP overhead)
- ✅ **Same as Claude** - Uses same approach as Claude Desktop
- ✅ **Better testing** - Tests actual MCP client functionality
- ✅ **More reliable** - No network issues with API endpoints

## Prerequisites

### 1. MongoDB Access on cmd.dev

**Check if MongoDB is accessible**:

```bash
# On cmd.dev, check MongoDB connection
mongosh "mongodb://localhost:27017/justjot" --eval "db.ideas.countDocuments()"
```

**Or check environment variables**:

```bash
echo $MONGODB_URI
# Should show: mongodb://localhost:27017/justjot
# Or: mongodb://mongo:27017/justjot (Docker service name)
```

### 2. MCP Server Path

**Check if MCP server is compiled**:

```bash
# On cmd.dev
ls -la /var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js
```

**If not compiled**:

```bash
cd /var/www/sites/personal/stock_market/JustJot.ai
npm run build:mcp
# or
tsc src/mcp/server.ts --outDir dist/mcp
```

### 3. Node.js Available

```bash
which node
node --version
```

## Configuration

### Option 1: Use MCP Client (Direct MongoDB)

**File**: `Jotty/core/integration/mcp_client.py`

```python
from core.integration.mcp_client import MCPClient

# On cmd.dev, MongoDB is accessible
client = MCPClient(
    server_path="/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js",
    mongodb_uri="mongodb://localhost:27017/justjot",  # or from env
    clerk_secret_key=os.getenv("CLERK_SECRET_KEY")
)

# Connect and use
await client.connect()
tools = await client.list_tools()
result = await client.call_tool("create_idea", {
    "title": "Test Idea",
    "description": "Testing MCP on cmd.dev"
})
```

**Benefits**:
- ✅ Direct MongoDB access
- ✅ Faster than HTTP API
- ✅ Same as Claude Desktop

### Option 2: Use HTTP API (Current)

**File**: `Jotty/skills/mcp-justjot/tools.py`

```python
# Uses /api/internal/ideas endpoints
# Works over network
# No MongoDB access needed
```

**Benefits**:
- ✅ Works from any server
- ✅ No MongoDB access needed
- ✅ Uses existing API infrastructure

## Testing on cmd.dev

### Test MCP Client Approach

**File**: `Jotty/test_mcp_cmd_dev.py`

```python
#!/usr/bin/env python3
"""Test MCP client on cmd.dev"""

import asyncio
import os
import sys
sys.path.insert(0, '/var/www/sites/personal/stock_market/Jotty')

from core.integration.mcp_client import MCPClient

async def test_mcp_on_cmd_dev():
    print("🧪 Testing MCP Client on cmd.dev...")
    print()
    
    # Get MongoDB URI from environment or use default
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/justjot")
    clerk_secret_key = os.getenv("CLERK_SECRET_KEY", "")
    
    print(f"   MongoDB URI: {mongodb_uri}")
    print(f"   MCP Server: /var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js")
    print()
    
    client = MCPClient(
        server_path="/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js",
        mongodb_uri=mongodb_uri,
        clerk_secret_key=clerk_secret_key
    )
    
    try:
        print("🔌 Connecting to MCP server...")
        await client.connect()
        print("✅ Connected!")
        print()
        
        print("📋 Listing tools...")
        tools = await client.list_tools()
        print(f"✅ Found {len(tools)} tools")
        print()
        
        print("🚀 Creating test idea...")
        result = await client.call_tool("create_idea", {
            "title": "Test Idea from MCP on cmd.dev",
            "description": "Testing MCP client with direct MongoDB access",
            "tags": ["test", "mcp", "cmd.dev"]
        })
        
        print("\n📊 Result:")
        print(result)
        print()
        
        if "success" in result and result["success"]:
            idea_id = result.get("id")
            print(f"✅ Idea created successfully!")
            print(f"   Idea ID: {idea_id}")
            
            # Test getting the idea
            print("\n📖 Getting idea...")
            get_result = await client.call_tool("get_idea", {"id": idea_id})
            print(f"✅ Retrieved idea: {get_result.get('title', 'N/A')}")
        else:
            print("❌ Failed to create idea")
            print(f"   Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 Disconnecting...")
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_mcp_on_cmd_dev())
```

**Run**:

```bash
cd /var/www/sites/personal/stock_market/Jotty
python3 test_mcp_cmd_dev.py
```

### Test HTTP API Approach

**File**: `Jotty/test_http_api_cmd_dev.py`

```python
#!/usr/bin/env python3
"""Test HTTP API on cmd.dev"""

import asyncio
import sys
sys.path.insert(0, '/var/www/sites/personal/stock_market/Jotty')

from core.registry.skills_registry import get_skills_registry

async def test_http_api_on_cmd_dev():
    print("🧪 Testing HTTP API on cmd.dev...")
    print()
    
    registry = get_skills_registry()
    registry.init()
    
    mcp_skill = registry.get_skill('mcp-justjot')
    if mcp_skill:
        create_tool = mcp_skill.tools.get('create_idea_tool')
        if create_tool:
            print("🚀 Creating idea via HTTP API...")
            result = await create_tool({
                'title': 'Test Idea via HTTP API on cmd.dev',
                'description': 'Testing HTTP API approach',
                'tags': ['test', 'http-api', 'cmd.dev']
            })
            
            print("\n📊 Result:")
            print(f"   Success: {result.get('success')}")
            if result.get('success'):
                print(f"   ✅ Idea ID: {result.get('id')}")
            else:
                print(f"   ⚠️  Error: {result.get('error')}")
    else:
        print("❌ Skill not found")

if __name__ == "__main__":
    asyncio.run(test_http_api_on_cmd_dev())
```

## Comparison: MCP Client vs HTTP API on cmd.dev

### MCP Client (Direct MongoDB)

**Pros**:
- ✅ Faster (direct DB access)
- ✅ Same as Claude Desktop
- ✅ Better for testing MCP functionality
- ✅ No HTTP overhead

**Cons**:
- ❌ Requires MongoDB access
- ❌ Requires Node.js and compiled MCP server
- ❌ Only works on cmd.dev (where MongoDB is accessible)

### HTTP API

**Pros**:
- ✅ Works from any server
- ✅ No MongoDB access needed
- ✅ Uses existing API infrastructure
- ✅ Works over network

**Cons**:
- ❌ HTTP overhead
- ❌ Requires API server running
- ❌ Network latency

## Recommendation

**For testing on cmd.dev**: Use **MCP Client** approach

**Why**:
1. cmd.dev has MongoDB accessible
2. Faster and more direct
3. Tests actual MCP client functionality
4. Same as Claude Desktop (better comparison)

**For production/deployment**: Use **HTTP API** approach

**Why**:
1. Works from any server
2. No MongoDB access needed
3. More flexible deployment

## Implementation

### Update `mcp-justjot-mcp-client` Skill

**File**: `Jotty/skills/mcp-justjot-mcp-client/tools.py`

```python
import os
from core.integration.mcp_client import MCPClient

# Detect if we're on cmd.dev (MongoDB accessible)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/justjot")
SERVER_PATH = "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"

# Check if MongoDB is accessible
def _is_mongodb_accessible():
    try:
        import pymongo
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return True
    except:
        return False

# Use MCP client if MongoDB accessible, else fallback to HTTP API
if _is_mongodb_accessible():
    # Use MCP client (direct MongoDB)
    client = MCPClient(
        server_path=SERVER_PATH,
        mongodb_uri=MONGODB_URI,
        clerk_secret_key=os.getenv("CLERK_SECRET_KEY", "")
    )
else:
    # Fallback to HTTP API
    client = None  # Use HTTP API skill instead
```

## Next Steps

1. **Check MongoDB on cmd.dev**:
   ```bash
   mongosh "mongodb://localhost:27017/justjot" --eval "db.ideas.countDocuments()"
   ```

2. **Compile MCP server** (if needed):
   ```bash
   cd JustJot.ai
   npm run build:mcp
   ```

3. **Test MCP client**:
   ```bash
   cd Jotty
   python3 test_mcp_cmd_dev.py
   ```

4. **Compare performance**:
   - MCP client: Direct MongoDB
   - HTTP API: Via `/api/internal/ideas`

5. **Choose approach**:
   - cmd.dev testing: MCP client
   - Production: HTTP API
