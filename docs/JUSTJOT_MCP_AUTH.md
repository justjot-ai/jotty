# JustJot.ai MCP HTTP Authentication Guide

## Overview

Since JustJot.ai uses Clerk with Google OAuth (browser-based), Jotty needs a way to authenticate for service-to-service calls.

## Authentication Methods

### Method 1: API Key + User ID (Recommended for Jotty)

**Best for:** Service-to-service authentication, scripts, automated systems

**How it works:**
- Uses `CLERK_SECRET_KEY` as API key
- Provides `userId` (Clerk user ID) in headers
- Server verifies using Clerk backend API

**Configuration:**
```bash
# In Jotty .env file
JUSTJOT_API_URL=https://justjot.ai
JUSTJOT_API_KEY=sk_test_...  # CLERK_SECRET_KEY
JUSTJOT_USER_ID=user_36W0zSjAkJe54fkRtDWLb4qMrpH  # Your Clerk user ID
```

**Usage:**
```python
from skills.justjot_mcp_http.tools import get_client

client = get_client()  # Uses env vars automatically
result = await client.call_tool('create_idea', {'title': 'Test'})
```

**Headers sent:**
```
x-api-key: sk_test_...
x-user-id: user_36W0zSjAkJe54fkRtDWLb4qMrpH
```

### Method 2: Bearer Token (Browser Session)

**Best for:** User-initiated actions, browser-based integrations

**How it works:**
- User logs in via browser (Google OAuth)
- Extract session token from browser
- Pass token to Jotty

**Getting the token:**
1. Log in to https://justjot.ai
2. Open browser console (F12)
3. Run: `await window.Clerk?.session?.getToken()`
4. Copy the token

**Configuration:**
```bash
JUSTJOT_AUTH_TOKEN=<session_token_from_browser>
```

**Usage:**
```python
client = JustJotMCPHTTPClient(auth_token='<token>')
```

**Headers sent:**
```
Authorization: Bearer <session_token>
```

## Security Considerations

### API Key Method (Method 1)

**Pros:**
- ✅ No user interaction needed
- ✅ Works for automated systems
- ✅ Uses existing Clerk infrastructure

**Cons:**
- ⚠️ Requires `CLERK_SECRET_KEY` (keep secure!)
- ⚠️ Need to know `userId` upfront

**Security:**
- Store `CLERK_SECRET_KEY` securely (env vars, secrets manager)
- Never commit to git
- Rotate keys periodically
- Use different keys for dev/prod

### Bearer Token Method (Method 2)

**Pros:**
- ✅ Uses existing user session
- ✅ No additional secrets needed
- ✅ User controls access

**Cons:**
- ⚠️ Requires user to log in first
- ⚠️ Tokens expire (need refresh)
- ⚠️ Not ideal for automated systems

## Finding Your Clerk User ID

### Option 1: Browser Console
```javascript
// In browser console on https://justjot.ai
await window.Clerk?.user?.id
// Returns: "user_36W0zSjAkJe54fkRtDWLb4qMrpH"
```

### Option 2: API Response
Check any API response that includes user info.

### Option 3: Database Query
```python
# Query MongoDB for your ideas
# userId field contains your Clerk user ID
```

## Implementation Details

### Server-Side (JustJot.ai API)

The API routes check authentication in this order:

1. **API Key + User ID** (`x-api-key` + `x-user-id` headers)
   - Verifies `x-api-key` matches `CLERK_SECRET_KEY`
   - Verifies user exists in Clerk
   - Uses `x-user-id` as `authId`

2. **Bearer Token** (`Authorization: Bearer <token>`)
   - Verifies token with Clerk
   - Extracts `userId` from token
   - Uses as `authId`

3. **Clerk Session** (via `requireAuth`)
   - Standard browser-based auth
   - Falls back if no API key/token

### Client-Side (Jotty)

The `JustJotMCPHTTPClient` automatically:
- Checks for `JUSTJOT_API_KEY` + `JUSTJOT_USER_ID` (Method 1)
- Falls back to `JUSTJOT_AUTH_TOKEN` (Method 2)
- Sends appropriate headers

## Example: Complete Setup

### 1. Get Your Clerk User ID
```bash
# Log in to https://justjot.ai
# Open browser console, run:
# await window.Clerk?.user?.id
# Copy the result
```

### 2. Configure Jotty
```bash
# Edit Jotty/.env
JUSTJOT_API_URL=https://justjot.ai
JUSTJOT_API_KEY=sk_test_f3w8tufFj1bW1DeHzJSVlCS0ljEJzMNxAuM2DpsSl4
JUSTJOT_USER_ID=user_36W0zSjAkJe54fkRtDWLb4qMrpH
```

### 3. Use in Jotty Skills
```python
from skills.justjot_mcp_http.tools import create_idea_tool

result = await create_idea_tool({
    'title': 'My Idea',
    'description': 'Created via Jotty',
    'tags': ['jotty', 'automated']
})
```

## Troubleshooting

### "Authentication required"
- Check `JUSTJOT_API_KEY` is set correctly
- Check `JUSTJOT_USER_ID` is your actual Clerk user ID
- Verify `CLERK_SECRET_KEY` matches on server

### "Invalid user ID"
- Verify user exists in Clerk
- Check user ID format: `user_xxx` (not MongoDB ObjectId)

### "Invalid token"
- Token may have expired
- Get new token from browser
- Or use API key method instead

## Best Practices

1. **For Production:**
   - Use environment variables (never hardcode)
   - Store secrets in secure vault
   - Use different keys for dev/staging/prod

2. **For Development:**
   - Use `.env` file (gitignored)
   - Test with your own user ID first
   - Verify API key matches server

3. **For Automation:**
   - Prefer API Key method (no user interaction)
   - Store user ID in config
   - Handle errors gracefully
