# How to Get Your Clerk User ID

## Quick Method (Browser Console)

1. **Log into JustJot.ai**: https://justjot.ai/dashboard
2. **Open Browser Console**: Press `F12` → Go to "Console" tab
3. **Copy and paste this**:

```javascript
window.Clerk?.user?.id || window.__clerk?.user?.id
```

4. **Press Enter** - it will show your Clerk user ID (starts with `user_`)

## Alternative: Check Network Tab

1. **Open DevTools**: `F12`
2. **Go to Network tab**
3. **Refresh dashboard** (`Ctrl+R` or `Cmd+R`)
4. **Find API call** to `/api/ideas` or `/api/dashboard`
5. **Click on it** → Check **Response** tab
6. **Look for** `userId` field (starts with `user_`)

## Format

Clerk user IDs look like:
- ✅ `user_36W0zSjAkJe54fkRtDWLb4qMrpH` (correct format)
- ✅ `user_2abc123def456ghi789` (correct format)
- ❌ `697765723425c87b977930b5` (MongoDB ObjectId, NOT Clerk ID)

## Once You Have It

Share your Clerk user ID (should start with `user_`) and I'll update the code to use it!
