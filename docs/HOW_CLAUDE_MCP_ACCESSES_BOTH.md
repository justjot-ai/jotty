# How Claude MCP Accesses Both MongoDB and API

## Overview

Claude's MCP server for JustJot.ai uses **direct MongoDB connection** for all operations. It does NOT use HTTP API endpoints.

## MCP Configuration

**File**: `JustJot.ai/mcp_config_claude.json`

```json
{
  "mcpServers": {
    "justjot": {
      "command": "node",
      "args": [
        "/path/to/JustJot.ai/src/mcp/server.ts"
      ],
      "env": {
        "MONGODB_URI": "mongodb://localhost:27017/justjot",
        "NEXT_PUBLIC_API_URL": "http://localhost:3000"
      }
    }
  }
}
```

## How It Works

### 1. MCP Server Startup

**File**: `JustJot.ai/src/mcp/server.ts`

```typescript
// Line 1355-1361
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('JustJot MCP Server running on stdio');
}

main().catch(console.error);
```

**Transport**: Uses `StdioServerTransport` (stdin/stdout)
- Claude Desktop spawns the Node.js process
- Communicates via JSON-RPC over stdio
- No HTTP involved

### 2. MongoDB Connection

**File**: `JustJot.ai/src/mcp/server.ts`

```typescript
// Server initialization (around line 50-100)
import mongoose from 'mongoose';

// MongoDB connection happens when tools are called
// Uses MONGODB_URI from environment
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/justjot';

// Connection is established via mongoose.connect() when needed
```

**Direct Connection**:
- Reads `MONGODB_URI` from environment
- Connects directly to MongoDB using `mongoose.connect()`
- No HTTP API calls to JustJot.ai backend

### 3. Tool Implementation Example: `create_idea`

**File**: `JustJot.ai/src/mcp/server.ts` (around line 400-600)

```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === 'create_idea') {
    // Connect to MongoDB directly
    await connectToDatabase(); // Uses mongoose.connect(MONGODB_URI)
    
    // Create idea directly in MongoDB
    const idea = new Idea({
      title: args.title,
      description: args.description,
      userId: args.userId || 'default-user',
      // ... other fields
    });
    
    await idea.save(); // Direct MongoDB write
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({ id: idea._id, ...idea.toObject() })
        }
      ]
    };
  }
});
```

**Key Points**:
- ✅ Direct MongoDB connection via `mongoose.connect(MONGODB_URI)`
- ✅ Direct database operations (`idea.save()`, `Idea.find()`, etc.)
- ❌ NO HTTP API calls to `/api/ideas`
- ❌ NO `fetch()` or `axios` calls

### 4. Database Connection Helper

**File**: `JustJot.ai/src/lib/mongodb.ts`

```typescript
import mongoose from 'mongoose';

const MONGODB_URI = process.env.MONGODB_URI!;

let cached = global.mongoose;

if (!cached) {
  cached = global.mongoose = { conn: null, promise: null };
}

async function connectToDatabase() {
  if (cached.conn) {
    return cached.conn;
  }

  if (!cached.promise) {
    cached.promise = mongoose.connect(MONGODB_URI).then((mongoose) => {
      return mongoose;
    });
  }

  cached.conn = await cached.promise;
  return cached.conn;
}
```

**How MCP Uses It**:
- MCP server calls `connectToDatabase()` before each tool execution
- Establishes direct MongoDB connection
- Uses connection pooling (cached connection)

## Why `NEXT_PUBLIC_API_URL` is in Config

**Question**: If MCP doesn't use HTTP API, why is `NEXT_PUBLIC_API_URL` in the config?

**Answer**: It's **NOT used by MCP server**. It's:
- Legacy/unused environment variable
- Or used by other parts of the codebase (frontend, API routes)
- MCP server only uses `MONGODB_URI`

## Comparison: MCP vs HTTP API

### MCP Server (Claude's Approach)

```
┌─────────────┐     stdio      ┌──────────────────┐     Direct     ┌──────────┐
│  Claude     │ ────────────> │  MCP Server      │ ────────────> │ MongoDB  │
│  Desktop    │   JSON-RPC    │  (Node.js)       │   mongoose    │          │
└─────────────┘               └──────────────────┘               └──────────┘
```

**Pros**:
- ✅ Direct database access (faster)
- ✅ No HTTP overhead
- ✅ Works offline (if MongoDB is local)
- ✅ No authentication needed (direct DB access)

**Cons**:
- ❌ Requires MongoDB access
- ❌ Can't work over network (stdio transport)
- ❌ Requires MongoDB running locally or accessible

### HTTP API (Jotty's Approach)

```
┌─────────────┐     HTTP       ┌──────────────────┐     Direct     ┌──────────┐
│  Jotty      │ ────────────> │  JustJot.ai API   │ ────────────> │ MongoDB  │
│  (Python)   │   REST API    │  (Next.js)        │   mongoose    │          │
└─────────────┘               └──────────────────┘               └──────────┘
```

**Pros**:
- ✅ Works over network (cmd.dev)
- ✅ No MongoDB access needed
- ✅ Uses existing API infrastructure
- ✅ Can use internal endpoints (`/api/internal/*`)

**Cons**:
- ❌ HTTP overhead
- ❌ Requires API server running
- ❌ May need authentication (unless using internal endpoints)

## Summary

**Claude MCP**:
- Uses **direct MongoDB connection** only
- Does NOT use HTTP API
- `NEXT_PUBLIC_API_URL` is unused by MCP server
- Works via stdio transport (local subprocess)

**Jotty MCP Skill**:
- Uses **HTTP API** (`/api/internal/ideas`)
- Does NOT need MongoDB access
- Works over network (cmd.dev)
- Uses `x-internal-service` header for auth bypass

Both approaches work, but they use different access methods!
