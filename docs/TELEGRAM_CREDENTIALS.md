# Telegram Credentials Used in Jotty Skills

## Credentials Source

Jotty skills use Telegram credentials from **environment variables** with fallback defaults.

## Default Credentials (from scripts)

**File**: `Jotty/scripts/v2v_trending_to_pdf.sh`

```bash
export TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-5228780618:AAE1W6XghhgnFtOGsUJfee_NRUssx32RyOU}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-810015653}"
```

**Credentials**:
- **Token**: `5228780618:AAE1W6XghhgnFtOGsUJfee_NRUssx32RyOU`
- **Chat ID**: `810015653`

## How Skills Get Credentials

**File**: `Jotty/skills/telegram-sender/tools.py`

```python
# Get credentials from params or environment
token = params.get('token') or os.getenv('TELEGRAM_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = params.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
```

**Priority Order**:
1. **Function parameter** (`params.get('token')` or `params.get('chat_id')`)
2. **Environment variable** (`TELEGRAM_TOKEN` or `TELEGRAM_CHAT_ID`)
3. **Fallback env var** (`TELEGRAM_BOT_TOKEN` for token)
4. **Default values** (from scripts, if env vars not set)

## Environment Variables

**Required**:
- `TELEGRAM_TOKEN` - Bot token (or `TELEGRAM_BOT_TOKEN`)
- `TELEGRAM_CHAT_ID` - Chat/channel ID

**Current Status**:
- ❌ Not set in environment (checked via `env | grep -i telegram`)
- ✅ Uses defaults from scripts if not set

## Usage in Skills

All Jotty skills that send to Telegram use the `telegram-sender` skill:

**Example**: `last30days-to-pdf-telegram/tools.py`

```python
telegram_skill = registry.get_skill('telegram-sender')
send_file_tool = telegram_skill.tools.get('send_telegram_file_tool')

telegram_result = await send_file_tool({
    'file_path': pdf_path,
    'chat_id': telegram_chat_id,  # Optional - uses env var if None
    'caption': 'Research PDF'
})
```

## Skills Using Telegram

1. `telegram-sender` - Core Telegram sending skill
2. `last30days-to-pdf-telegram` - Research → PDF → Telegram
3. `last30days-to-epub-telegram` - Research → EPUB → Telegram
4. `search-summarize-pdf-telegram` - Search → Summarize → PDF → Telegram
5. `v2v-to-pdf-telegram-remarkable` - V2V → PDF → Telegram + reMarkable
6. `screener-to-pdf-telegram` - Screener → PDF → Telegram

## Summary

**Credentials Used**:
- **Token**: `5228780618:AAE1W6XghhgnFtOGsUJfee_NRUssx32RyOU`
- **Chat ID**: `810015653`

**Source**:
- Default values in `v2v_trending_to_pdf.sh` script
- Can be overridden via environment variables
- Can be passed as function parameters

**Note**: These credentials are hardcoded defaults in the script. For production, set them as environment variables instead.
