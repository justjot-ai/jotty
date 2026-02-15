# How to Run Telegram Bot (Migrated)

## Quick Start

```bash
# Start the bot
python -m apps.telegram.bot_migrated

# You should see:
# INFO - Telegram bot initialized with shared components
# INFO - Starting Telegram bot...
# INFO - Application started
```

## If Commands Don't Respond

The bot might be running but not responding to commands. Here's how to debug:

### 1. Check Bot is Running

Look for these log messages:
```
INFO - Telegram bot initialized with shared components
INFO - Starting Telegram bot...
```

### 2. Check for Errors

If you see errors like:
```
TypeError: object NoneType can't be used in 'await' expression
```

This means the `send_callback` function is not returning a coroutine properly.

### 3. The Real Issue

Looking at the code, I found the problem! The `send_callback` in `_handle_command` doesn't return anything, so `await send_callback()` causes an error.

## FIX: Don't Await send_callback in Commands

The send_callback is already async in the real bot context, but it doesn't need to be awaited in every command. Let me fix this.
