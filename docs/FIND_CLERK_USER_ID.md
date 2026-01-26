# How to Find Your Clerk User ID

## Problem

Ideas created with `userId: 'mcp-client-user'` won't show up in your dashboard because that's not your actual Clerk user ID.

## Solution

You need to provide your actual Clerk user ID when creating ideas.

## Methods to Find Your Clerk User ID

### Method 1: From JustJot.ai Dashboard URL

1. Log into https://justjot.ai/dashboard
2. Open browser developer tools (F12)
3. Check the Network tab
4. Look for API calls - your user ID will be in the requests
5. Or check localStorage: `localStorage.getItem('clerk-session')`

### Method 2: From Clerk Dashboard

1. Go to https://dashboard.clerk.com
2. Navigate to Users
3. Find your user (setia.naveen@gmail.com)
4. Copy the User ID (starts with `user_`)

### Method 3: Using Clerk API

```bash
curl -H "Authorization: Bearer YOUR_CLERK_SECRET_KEY" \
  https://api.clerk.com/v1/users?email_address=setia.naveen@gmail.com
```

### Method 4: Check Existing Ideas in MongoDB

```python
import pymongo

client = pymongo.MongoClient("mongodb://...")
db = client['planmyinvesting']
ideas = db.ideas.find({"title": {"$exists": True}}).limit(1)

for idea in ideas:
    print(f"User ID: {idea.get('userId')}")
```

## Using Your User ID

Once you have your Clerk user ID, use it when creating ideas:

```python
result = await tool({
    'topic': 'multi-agent systems',
    'userId': 'user_2abc123xyz...',  # Your actual Clerk user ID
    'use_mcp_client': True
})
```

## Format

Clerk user IDs typically look like:
- `user_2abc123def456ghi789`
- Starts with `user_`
- Followed by alphanumeric characters

## Quick Fix

If you can't find your user ID immediately, you can:

1. Create an idea manually in JustJot.ai dashboard
2. Check the idea's `userId` field in MongoDB
3. Use that userId for future MCP-created ideas
