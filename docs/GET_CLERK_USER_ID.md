# How to Get Your Clerk User ID

## Problem

Ideas need to be assigned to your Clerk user ID to appear in your dashboard. Clerk user IDs start with `user_` (e.g., `user_36W0zSjAkJe54fkRtDWLb4qMrpH`).

## Quick Method: Browser Console

1. **Log into JustJot.ai**: https://justjot.ai/dashboard
2. **Open Browser Console**: Press `F12` or `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Option+I` (Mac)
3. **Go to Console tab**
4. **Type one of these**:
   ```javascript
   window.Clerk?.user?.id
   // or
   window.__clerk?.user?.id
   // or
   Clerk?.user?.id
   ```
5. **Copy the user ID** (should start with `user_`)

## Alternative: Network Tab

1. **Open DevTools**: `F12`
2. **Go to Network tab**
3. **Refresh the dashboard**
4. **Look for API calls** to `/api/ideas` or `/api/dashboard`
5. **Click on a request**
6. **Check Response** or **Request Headers** for `userId` or `user_id`

## Alternative: Check Existing Idea

If you have an idea that shows in your dashboard:

1. **Open the idea** in JustJot.ai
2. **Check the URL**: `https://justjot.ai/dashboard/ideas/{idea_id}`
3. **Open browser console** and check the idea data:
   ```javascript
   // The idea object might be in React state
   // Or check Network tab for the API response
   ```

## Format

Clerk user IDs look like:
- ✅ `user_36W0zSjAkJe54fkRtDWLb4qMrpH` (correct format)
- ❌ `697765723425c87b977930b5` (MongoDB ObjectId, not Clerk ID)

## Once You Have It

Use it when creating ideas:

```python
result = await tool({
    'topic': 'multi-agent systems',
    'userId': 'user_YOUR_CLERK_USER_ID_HERE',  # Your Clerk user ID
    'use_mcp_client': True
})
```
