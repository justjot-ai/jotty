# Environment Configuration (.env)

Jotty skills can load shared credentials from a `.env` file located in the Jotty root directory.

## Setup

1. **Copy the example file:**
   ```bash
   cp Jotty/.env.example Jotty/.env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   nano Jotty/.env
   ```

3. **Add your credentials:**
   ```env
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   ```

## How It Works

- The `.env` file is automatically loaded when `SkillsRegistry` initializes
- Skills use `os.getenv()` to read environment variables
- The `.env` file is gitignored (not committed to version control)
- Environment variables take priority: function params > existing env vars > .env file

## Current Credentials

Telegram credentials are stored in `.env`:
- **Token**: From `v2v_trending_to_pdf.sh` (planmyinvesting.com)
- **Chat ID**: From `v2v_trending_to_pdf.sh`

## Adding More Credentials

To add more shared credentials:

1. Add to `.env`:
   ```env
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

2. Skills can access them via:
   ```python
   import os
   api_key = os.getenv('OPENAI_API_KEY')
   ```

## Security Notes

- ✅ `.env` is gitignored (not committed)
- ✅ `.env.example` is committed (template only)
- ✅ Never commit actual credentials
- ✅ Use different credentials for development/production

## Skills Using .env

All skills that need credentials automatically use `.env`:

- **telegram-sender**: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
- Future skills can add their own credentials to `.env`

## Testing

Test that `.env` is loaded:

```python
from core.registry.skills_registry import get_skills_registry
import os

registry = get_skills_registry()
registry.init()

token = os.getenv('TELEGRAM_TOKEN')
print(f"Token loaded: {bool(token)}")
```
