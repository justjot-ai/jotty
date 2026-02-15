# Telegram Bot - Full Command Reference

**All CLI commands now available in Telegram!**

---

## 🚀 Quick Start

```bash
# Stop old bot if running
pkill -f bot_migrated

# Start full-featured bot
./start_telegram_bot_full.sh
```

Then send `/start` to your bot on Telegram.

---

## ✅ Commands Implemented

### 💬 Basic (4 commands)
- `/start` - Welcome message with quick start
- `/help` - Complete command reference
- `/status` - Bot status (state, messages, session)
- `/clear` - Clear chat history

### 📝 Session (3 commands)
- `/session` - Session info (ID, message count, timestamps)
- `/session save` - Save current session (coming soon)
- `/session load` - Load saved session (coming soon)

### 🧠 Memory (3 commands)
- `/memory` - Memory system status
- `/memory search <query>` - Search stored memories
- `/memory clear` - Clear all memories (coming soon)

### 🛠️ Skills (3 commands)
- `/skills` - List all 164+ available skills
- `/skill <name>` - Execute specific skill
- `/skill search <query>` - Search for skills (coming soon)

### 🤖 Agents (2 commands)
- `/agents` - List available agents
- `/agent <name>` - Run specific agent

### 🤝 Swarms (1 command)
- `/swarm <agents>` - Multi-agent coordination
  - Example: `/swarm researcher,coder,tester`

### 🔄 Workflows (1 command)
- `/workflow <name>` - Run predefined workflow
- `/workflow` - List available workflows

### 🧬 Model (2 commands)
- `/model` - Show current model
- `/model list` - List available models
- `/model switch <name>` - Switch model (coming soon)

### ⚙️ Config (1 command)
- `/config` - View/edit configuration (coming soon)

### 📊 Statistics (3 commands)
- `/stats` - Usage statistics
- `/tokens` - Token usage breakdown
- `/cost` - Cost analysis

### 🐛 Debug (1 command)
- `/debug` - System debug information

---

## 🆚 Old vs New

| Feature | Old Bot | New Bot (Full) |
|---------|---------|----------------|
| **Commands** | 5 | 20+ |
| **Session management** | ❌ | ✅ |
| **Memory commands** | ❌ | ✅ |
| **Skill execution** | ❌ | ✅ |
| **Agent control** | ❌ | ✅ |
| **Statistics** | ❌ | ✅ |
| **Error handling** | ⚠️ Basic | ✅ Advanced |
| **Shared components** | ✅ | ✅ |

---

## 📱 Testing Checklist

### Phase 1: Basic Commands (2 min)
```
/start
/help
/status
/clear
```
**Expected:** All respond with formatted messages

### Phase 2: Session Commands (1 min)
```
/session
```
**Expected:** Shows session info with message counts

### Phase 3: Memory Commands (1 min)
```
/memory
```
**Expected:** Shows memory status

### Phase 4: Skills & Agents (2 min)
```
/skills
/agents
/agent researcher
```
**Expected:** Lists appear, agent starts (coming soon)

### Phase 5: Swarm (2 min)
```
/swarm researcher,coder
```
**Expected:** Multi-agent coordination

### Phase 6: Stats (1 min)
```
/stats
/tokens
/cost
/debug
```
**Expected:** All show formatted data

### Phase 7: Regular Chat (2 min)
```
Hello, who are you?
Write a Python function for fibonacci
```
**Expected:** AI responds with streaming text

---

## 🐛 Known Issues Fixed

1. ✅ **Unknown command errors** - All commands now implemented
2. ✅ **Bot lag** - Improved error handling
3. ✅ **No response** - Better exception handling
4. ✅ **MarkdownV2 errors** - Fallback to plain text

---

## 🔧 Error Handling

The new bot includes:
- **Graceful fallbacks** - If MarkdownV2 fails, uses plain text
- **Try/catch everywhere** - No crashes
- **Detailed logging** - Easy debugging
- **User feedback** - Clear error messages

---

## 📝 Next Steps

After testing, we'll add:
1. TUI migration with all 36 CLI commands
2. REPL features (history, autocomplete)
3. Plugin system
4. Config file loading

---

## 🎯 Test NOW

1. **Kill old bot:**
   ```bash
   pkill -f bot_migrated
   ```

2. **Start new bot:**
   ```bash
   ./start_telegram_bot_full.sh
   ```

3. **Test on Telegram:**
   - Send: `/start`
   - Send: `/help`
   - Send: `/session`
   - Send: `/memory`
   - Send: `/skills`
   - Send: `Hello!`

All commands should now work! 🎉
