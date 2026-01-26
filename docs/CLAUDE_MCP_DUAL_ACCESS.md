# Claude MCP Dual Access: MongoDB + Clerk API

## Summary

Claude's MCP server accesses **TWO** external services:

1. **MongoDB** - Direct connection (via mongoose)
2. **Clerk API** - HTTP calls (for user lookup/validation)

## Configuration

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

**Both environment variables are used!**

## Access Method 1: MongoDB (Direct Connection)

### Code Location

**File**: `JustJot.ai/src/mcp/server.ts` (Line 134, 188-195)

```typescript
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/justjot';

async function connectDB() {
  if (mongoose.connection.readyState === 1) return;
  
  await mongoose.connect(MONGODB_URI, {
    bufferCommands: false,
    maxPoolSize: 5,
    serverSelectionTimeoutMS: 5000,
  });
}
```

### Usage Example: `create_idea`

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 612-660)

```typescript
case 'create_idea': {
  await connectDB();  // Direct MongoDB connection
  
  const newIdea = new Idea({
    title: args.title,
    description: args.description || '',
    // ... other fields
    userId: args.userId || null,  // Can be null!
  });
  
  await newIdea.save();  // Direct MongoDB write
  
  return JSON.stringify({
    success: true,
    id: newIdea._id,
    message: `Idea "${args.title}" created successfully`
  });
}
```

**Access Pattern**:
- ✅ Direct connection via `mongoose.connect(MONGODB_URI)`
- ✅ All CRUD operations use Mongoose models directly
- ✅ No HTTP API calls to JustJot.ai backend

## Access Method 2: Clerk API (HTTP Calls)

### Code Location

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 135, 150-185)

```typescript
const CLERK_SECRET_KEY = process.env.CLERK_SECRET_KEY || '';

interface ClerkUser {
  id: string;
  emailAddresses: Array<{ emailAddress: string }>;
  firstName?: string;
  lastName?: string;
}

async function fetchClerkUsers(userId?: string): Promise<ClerkUser[]> {
  return new Promise((resolve, reject) => {
    if (!CLERK_SECRET_KEY) {
      reject(new Error('CLERK_SECRET_KEY not configured'));
      return;
    }

    const path = userId
      ? `/v1/users/${userId}`
      : '/v1/users';

    const req = https.request({
      hostname: 'api.clerk.com',
      path: path,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${CLERK_SECRET_KEY}` },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.errors) {
            reject(new Error(parsed.errors[0]?.message || 'Clerk API error'));
          } else {
            resolve(userId ? [parsed] : parsed);
          }
        } catch {
          reject(new Error('Failed to parse Clerk response'));
        }
      });
    }).on('error', reject);
    
    req.end();
  });
}
```

**Access Pattern**:
- ✅ HTTP calls to `api.clerk.com`
- ✅ Uses `CLERK_SECRET_KEY` for authentication
- ✅ Uses Node.js `https` module (not fetch/axios)

### Usage Example: `assign_idea` Tool

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 680-695)

```typescript
case 'assign_idea': {
  await connectDB();  // MongoDB connection
  
  // Validate user exists via Clerk API
  try {
    const clerkUsers = await fetchClerkUsers(args.userId as string | undefined);
    
    if (clerkUsers.length === 0) {
      return JSON.stringify({
        error: `User ${args.userId} not found in Clerk`,
        hint: 'Ensure CLERK_SECRET_KEY environment variable is set correctly'
      });
    }
    
    const user = clerkUsers[0];
    const updateData: Record<string, unknown> = {
      userId: args.userId,
    };
    if (args.author) updateData.author = args.author;

    const updated = await Idea.findByIdAndUpdate(args.id, updateData, { new: true });
    if (!updated) return JSON.stringify({ error: 'Idea not found' });
    
    return JSON.stringify({
      success: true,
      message: `Idea "${updated.title}" assigned to user ${args.userId}`,
      ideaId: updated._id,
      userId: args.userId,
      author: updated.author,
    });
  } catch (error) {
    return JSON.stringify({
      error: error instanceof Error ? error.message : 'Failed to fetch user details',
      hint: 'Ensure CLERK_SECRET_KEY environment variable is set correctly'
    });
  }
}
```

**What happens**:
1. Connects to MongoDB
2. **Calls Clerk API** to validate user exists
3. Updates idea in MongoDB with validated userId

## Complete Access Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Claude Desktop                                               │
│  - Spawns: node dist/mcp/server.js                          │
│  - Communicates via: stdio (JSON-RPC)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ JSON-RPC over stdin/stdout
                        │
┌───────────────────────▼─────────────────────────────────────┐
│ MCP Server (Node.js)                                         │
│  - Reads: MONGODB_URI from env                               │
│  - Reads: CLERK_SECRET_KEY from env                          │
│  - Transport: StdioServerTransport                           │
└───────┬───────────────────────────────────────┬─────────────┘
        │                                       │
        │ mongoose.connect(MONGODB_URI)         │ https.request()
        │                                       │ api.clerk.com
        │                                       │
┌───────▼──────────┐                  ┌────────▼──────────────┐
│ MongoDB          │                  │ Clerk API              │
│  - Direct        │                  │  - User validation     │
│  - CRUD ops      │                  │  - User lookup        │
│  - No HTTP       │                  │  - Authentication     │
└──────────────────┘                  └───────────────────────┘
```

## When Each Access Method is Used

### MongoDB (Direct Connection)

**Used for**:
- ✅ Creating ideas (`create_idea`)
- ✅ Listing ideas (`list_ideas`)
- ✅ Getting idea (`get_idea`)
- ✅ Updating ideas (`update_idea`)
- ✅ Deleting ideas (`delete_idea`)
- ✅ Managing templates (`list_templates`, `create_template`, etc.)
- ✅ Managing tags (`list_tags`, `get_ideas_by_tag`)
- ✅ Managing sections (add, update, delete sections)

**Access**: Direct via `mongoose.connect()` → MongoDB

### Clerk API (HTTP Calls)

**Used for**:
- ✅ Validating user exists (`assign_idea` tool)
- ✅ Looking up user details (`assign_idea` tool)
- ✅ User authentication (if needed in future)

**Access**: HTTP GET → `https://api.clerk.com/v1/users/{userId}`

**Note**: Most tools don't use Clerk API - only `assign_idea` validates users!

## Why This Design?

### MongoDB Direct Access

**Benefits**:
- ✅ Fast (no HTTP overhead)
- ✅ Works offline (if MongoDB is local)
- ✅ Full query capabilities (Mongoose query builder)
- ✅ No API server needed for MCP operations

**Requirements**:
- MongoDB must be accessible (local, Docker, or remote)

### Clerk API Access

**Benefits**:
- ✅ Validates users exist before assigning ideas
- ✅ Gets user details (email, name) for display
- ✅ Centralized user management (Clerk handles auth)

**Requirements**:
- Internet connection (Clerk API is cloud service)
- `CLERK_SECRET_KEY` must be valid

## Comparison: MCP vs HTTP API Approach

### MCP Server (Claude's Approach)

```
MongoDB: Direct connection (mongoose.connect)
Clerk: HTTP API (https.request to api.clerk.com)
JustJot API: NOT USED
```

### HTTP API Skill (Jotty's Approach)

```
MongoDB: NOT ACCESSED (no direct connection)
Clerk: NOT ACCESSED (no user validation)
JustJot API: HTTP calls (/api/internal/ideas)
```

## Summary

**Claude MCP accesses**:
1. ✅ **MongoDB** - Direct connection for all data operations
2. ✅ **Clerk API** - HTTP calls for user validation (only in `assign_idea`)

**Claude MCP does NOT access**:
- ❌ JustJot.ai HTTP API (`/api/ideas` endpoints)
- ❌ Next.js API routes
- ❌ Any internal JustJot.ai services

**Why it works**:
- MongoDB is accessible (local, Docker, or remote)
- Clerk API is publicly accessible (cloud service)
- No need for JustJot.ai API server for MCP operations

**Why Jotty uses HTTP API**:
- Jotty runs on cmd.dev (different server)
- Can't access MongoDB directly (network/security)
- Uses `/api/internal/ideas` endpoints
- Works over network without MongoDB access
