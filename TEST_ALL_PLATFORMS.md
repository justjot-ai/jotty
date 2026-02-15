# Test All Platforms - Complete Guide

## Quick Test Commands

### 1. TUI (Terminal) ✅
```bash
python -m apps.cli.app_migrated
```

**What to try:**
```
/help
/status
/skills
Hello, who are you?
/clear
```

**Expected:**
- Welcome message appears
- Commands show output
- Chat works with AI
- All 36 commands available

**Exit:** Press Ctrl+C or Ctrl+D

---

### 2. Telegram 📱

**A. Start the bot:**
```bash
# Kill any old bot
pkill -f bot_migrated

# Start new bot
./start_telegram_bot_full.sh
```

**B. Test on Telegram app:**

Send to your bot:
```
/start
/help
/status
/session
/memory
/skills
Hello!
```

**Expected:**
- Bot responds to all commands
- No lag
- No "unknown command" errors
- Chat works

**Stop:** Press Ctrl+C in terminal

---

### 3. Web 🌐

**A. Start backend:**
```bash
# Terminal 1
python apps/web/backend/server.py
```

**Expected output:**
```
Starting Jotty Web Server...
WebSocket: ws://localhost:8000/ws
Web UI: http://localhost:8000
```

**B. Test backend:**
```bash
# Terminal 2
curl http://localhost:8000/health
```

**Expected:**
```json
{"status":"healthy","sessions":0,"version":"1.0.0"}
```

**C. Start frontend:**
```bash
# Terminal 2 (or new terminal)
cd apps/web/frontend

# First time only:
npm install

# Start dev server:
npm start
```

**Expected:**
```
Compiled successfully!
Local: http://localhost:3000
```

**D. Test in browser:**

1. Open: http://localhost:3000
2. You should see: "🤖 Jotty AI" header
3. Status: "✅ Connected | Session: xxxxxxxx"
4. Try typing:
   ```
   Hello!
   /help
   /status
   ```

**Expected:**
- Messages appear in chat
- Commands work
- Real-time updates
- No connection errors

**Stop:** Press Ctrl+C in both terminals

---

## Troubleshooting

### TUI Not Starting

**Error:** `ModuleNotFoundError`
```bash
# Make sure you're in the right directory
cd /var/www/sites/personal/stock_market/Jotty

# Try again
python -m apps.cli.app_migrated
```

**Error:** `Command failed`
- Some commands need SDK implementations
- This is normal - basic commands still work
- Try: `/help`, `/status`, `/clear`

---

### Telegram Not Responding

**Check if bot is running:**
```bash
ps aux | grep bot_migrated
```

**If not running:**
```bash
./start_telegram_bot_full.sh
```

**If still not responding:**

1. **Check logs:**
```bash
# Look for errors in the output
```

2. **Verify token:**
```bash
grep TELEGRAM_TOKEN .env
```

3. **Test connection:**
```bash
# Send /start to your bot on Telegram
# Wait 5 seconds
# Check terminal for logs
```

4. **Restart bot:**
```bash
pkill -f bot_migrated
./start_telegram_bot_full.sh
```

**Common issues:**
- Bot crashed due to error → Check logs, restart
- Network issue → Check internet connection
- Token invalid → Check .env file
- Rate limited → Wait a few minutes

---

### Web Not Connecting

**Backend issues:**

```bash
# Check if port 8000 is in use
lsof -i :8000

# If something else is using it, kill it:
kill -9 $(lsof -t -i:8000)

# Restart backend
python apps/web/backend/server.py
```

**Frontend issues:**

```bash
# Clear node modules
cd apps/web/frontend
rm -rf node_modules package-lock.json

# Reinstall
npm install

# Start
npm start
```

**WebSocket not connecting:**

1. **Check backend is running:**
   - Look for: "Starting Jotty Web Server..."

2. **Check browser console:**
   - F12 → Console
   - Look for WebSocket errors

3. **Test WebSocket directly:**
   ```bash
   # Install websocat: brew install websocat (Mac) or similar
   websocat ws://localhost:8000/ws

   # Should see: Connected message
   ```

4. **Check firewall:**
   - Make sure port 8000 is not blocked

---

## Quick Test Script

Save this as `test_all.sh`:

```bash
#!/bin/bash

echo "================================"
echo "Testing All Platforms"
echo "================================"
echo ""

# Test TUI
echo "1. Testing TUI..."
timeout 3 python -m apps.cli.app_migrated 2>&1 | head -5
if [ $? -eq 124 ]; then
    echo "✅ TUI starts successfully"
else
    echo "❌ TUI failed to start"
fi
echo ""

# Test Telegram
echo "2. Checking Telegram bot..."
if ps aux | grep -q "[b]ot_migrated"; then
    echo "✅ Telegram bot is running"
else
    echo "❌ Telegram bot is NOT running"
    echo "   Start with: ./start_telegram_bot_full.sh"
fi
echo ""

# Test Web backend
echo "3. Testing Web backend..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Web backend is running"
    curl -s http://localhost:8000/health | python -m json.tool
else
    echo "❌ Web backend is NOT running"
    echo "   Start with: python apps/web/backend/server.py"
fi
echo ""

echo "================================"
echo "Test Complete"
echo "================================"
```

**Run it:**
```bash
chmod +x test_all.sh
./test_all.sh
```

---

## Manual Testing Checklist

### TUI
- [ ] Starts without errors
- [ ] Shows welcome message
- [ ] `/help` works
- [ ] Chat works ("Hello!")
- [ ] Commands execute
- [ ] Ctrl+C exits cleanly

### Telegram
- [ ] Bot responds to `/start`
- [ ] `/help` shows all commands
- [ ] Chat works
- [ ] Commands execute
- [ ] No lag
- [ ] No "unknown command" errors

### Web
- [ ] Backend starts
- [ ] Frontend connects
- [ ] Shows "Connected" status
- [ ] Chat works
- [ ] Commands work
- [ ] Real-time updates
- [ ] No WebSocket errors

---

## Performance Check

### Response Times

**TUI:**
- Command response: < 100ms
- Chat response: 1-3s

**Telegram:**
- Command response: < 500ms
- Chat response: 2-4s (first message slower)

**Web:**
- WebSocket connect: < 100ms
- Command response: < 200ms
- Chat response: 1-3s

If slower, check:
- Network connection
- SDK performance
- Server resources

---

## Next Steps

After testing all platforms:

1. ✅ **All work** → You're done! 🎉
2. ⚠️ **Some work** → Fix issues using troubleshooting guide
3. ❌ **None work** → Check installation:
   ```bash
   pip install -r requirements.txt
   cd apps/web/frontend && npm install
   ```

---

## Getting Help

If stuck, check:
1. This guide's troubleshooting section
2. Error messages in terminal
3. Browser console (F12) for web
4. Telegram bot logs

**Common fixes:**
- Restart the service
- Check dependencies installed
- Check ports not in use
- Check firewall/network

---

**Happy testing! 🚀**
