# Claude MCP Access Flow - Detailed Code Walkthrough

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

**Key Points**:
- ✅ `MONGODB_URI` - Direct MongoDB connection string
- ✅ `CLERK_SECRET_KEY` - For user authentication (if needed)
- ❌ **NO** `NEXT_PUBLIC_API_URL` or `API_URL` - Not used!

## Code Flow: How MCP Accesses MongoDB

### Step 1: Server Initialization

**File**: `JustJot.ai/src/mcp/server.ts` (Line 134)

```typescript
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/justjot';
```

**What happens**:
- Reads `MONGODB_URI` from environment (set by Claude config)
- Defaults to `localhost:27017` if not set

### Step 2: Database Connection Function

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 188-195)

```typescript
async function ensureDatabaseConnection() {
  if (mongoose.connection.readyState === 1) return;
  
  await mongoose.connect(MONGODB_URI, {
    bufferCommands: false,
    maxPoolSize: 10,
    minPoolSize: 2,
  });
}
```

**What happens**:
- Checks if already connected (`readyState === 1`)
- If not, calls `mongoose.connect(MONGODB_URI)`
- **Direct MongoDB connection** - no HTTP involved!

### Step 3: Tool Handler Registration

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 540-550)

```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  // Ensure database connection
  await ensureDatabaseConnection();
  
  // Handle tool calls...
});
```

**What happens**:
- Every tool call ensures MongoDB connection
- Uses cached connection if already connected
- No HTTP API calls here!

### Step 4: Example: `create_idea` Tool

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 612-660)

```typescript
case 'create_idea': {
  // Validate sections before creating
  const sections = (args.sections as IIdeaSection[]) || [];
  const validationWarnings: string[] = [];

  // ... validation code ...

  // Create idea directly in MongoDB
  const newIdea = new Idea({
    title: args.title,
    description: args.description || '',
    templateName: args.templateName || 'Blank',
    status: args.status || 'Draft',
    tags: args.tags || [],
    userId: args.userId || null,  // Can be null - no auth required!
    author: args.author || 'You',
    sections: sections.map(s => ({
      title: s.title,
      content: s.content || '',
      type: s.type || 'text',
    })),
  });
  
  // Direct MongoDB save - NO HTTP API!
  await newIdea.save();

  return JSON.stringify({
    success: true,
    id: newIdea._id,
    message: `Idea "${args.title}" created successfully`
  });
}
```

**Key Points**:
- ✅ Creates `Idea` model instance directly
- ✅ Calls `newIdea.save()` - **direct MongoDB write**
- ✅ No `fetch()` or `axios` calls
- ✅ No HTTP API endpoints (`/api/ideas`)
- ✅ `userId` can be `null` - no authentication required!

### Step 5: Example: `list_ideas` Tool

**File**: `JustJot.ai/src/mcp/server.ts` (Lines 564-590)

```typescript
case 'list_ideas': {
  const query: Record<string, unknown> = {};
  if (args.status) query.status = args.status;
  if (args.templateName) query.templateName = args.templateName;
  if (args.tag) query.tags = args.tag;

  // Direct MongoDB query - NO HTTP API!
  const ideas = await Idea.find(query)
    .select('title description status templateName tags updatedAt sections')
    .sort({ updatedAt: -1 })
    .skip(Number(args.skip) || 0)
    .limit(Number(args.limit) || 20)
    .lean();

  return JSON.stringify(ideas.map(idea => ({
    id: idea._id,
    title: idea.title,
    description: idea.description,
    status: idea.status,
    templateName: idea.templateName,
    tags: idea.tags,
    sectionCount: idea.sections?.length || 0,
    updatedAt: idea.updatedAt,
  })));
}
```

**Key Points**:
- ✅ Uses `Idea.find()` - **direct MongoDB query**
- ✅ Uses Mongoose query builder (`.select()`, `.sort()`, `.skip()`, `.limit()`)
- ✅ No HTTP API calls
- ✅ Returns JSON string directly

## Why `CLERK_SECRET_KEY` is in Config

**Question**: If MCP doesn't use HTTP API, why is `CLERK_SECRET_KEY` there?

**Answer**: It's **currently unused** in the MCP server code!

**Possible reasons**:
1. **Future use** - May be used for user validation later
2. **Legacy** - Leftover from when API was used
3. **Optional auth** - Could validate `userId` if provided

**Current behavior**: `userId` can be `null` or any string - no validation!

## Complete Access Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Claude Desktop                                               │
│  - Spawns: node dist/mcp/server.js                          │
│  - Communicates via: stdio (JSON-RPC)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ JSON-RPC over stdin/stdout
                        │
┌───────────────────────▼─────────────────────────────────────┐
│ MCP Server (Node.js)                                         │
│  - Reads: MONGODB_URI from env                               │
│  - Reads: CLERK_SECRET_KEY from env (unused)                 │
│  - Transport: StdioServerTransport                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ mongoose.connect(MONGODB_URI)
                        │
┌───────────────────────▼─────────────────────────────────────┐
│ MongoDB                                                     │
│  - Host: localhost:27017 (or remote)                        │
│  - Database: justjot                                        │
│  - Collections: ideas, templates, sections, tags             │
└─────────────────────────────────────────────────────────────┘

❌ NO HTTP API CALLS
❌ NO /api/ideas endpoints
❌ NO fetch() or axios
✅ DIRECT MongoDB connection only!
```

## Comparison: MCP vs HTTP API

### MCP Server (Claude's Approach)

```typescript
// Direct MongoDB access
await ensureDatabaseConnection();
const idea = new Idea({ title: "Test", ... });
await idea.save();  // Direct write
```

**Access Method**: `mongoose.connect(MONGODB_URI)` → Direct MongoDB connection

### HTTP API (Jotty's Approach)

```python
# HTTP API call
response = requests.post(
    f"{API_URL}/api/internal/ideas",
    json={"title": "Test", ...},
    headers={"x-internal-service": "true"}
)
```

**Access Method**: HTTP POST → Next.js API route → MongoDB

## Summary

**Claude MCP**:
- ✅ Uses **direct MongoDB connection** (`mongoose.connect()`)
- ✅ Reads `MONGODB_URI` from environment
- ✅ All operations are direct database queries/writes
- ❌ Does **NOT** use HTTP API
- ❌ `CLERK_SECRET_KEY` is in config but **unused** in code
- ❌ `NEXT_PUBLIC_API_URL` is **NOT** in config (not needed!)

**Why it works**:
- MongoDB is accessible (local, Docker, or remote like MongoDB Atlas)
- MCP server connects directly via mongoose
- No HTTP server needed for MCP operations

**Why Jotty needs HTTP API**:
- Jotty runs on cmd.dev (different server)
- Can't access MongoDB directly (network/security)
- Uses HTTP API endpoints (`/api/internal/ideas`)
- Works over network without MongoDB access
