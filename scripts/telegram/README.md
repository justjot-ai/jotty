# Telegram Bot Scripts

Scripts for managing the Telegram bot.

## Scripts

### `start_telegram_bot.sh`
Start the Jotty Telegram bot.

```bash
./scripts/telegram/start_telegram_bot.sh
```

### `restart_telegram.sh`
Restart the Telegram bot (stops old instance and starts new one).

```bash
./scripts/telegram/restart_telegram.sh
```

## Usage

1. Set `TELEGRAM_TOKEN` in `.env`:
   ```bash
   TELEGRAM_TOKEN=your_bot_token_here
   ```

2. Start the bot:
   ```bash
   ./scripts/telegram/start_telegram_bot.sh
   ```

3. Send `/start` to your bot on Telegram

## Available Commands

- `/start` - Welcome message
- `/help` - Show help
- `/status` - Bot status
- `/clear` - Clear chat history
- `/session` - Session info
- `/memory` - Memory status
- `/skill` - Execute skill
- `/skills` - List skills
- `/agent` - Run agent
- `/agents` - List agents
- `/swarm` - Swarm coordination
- `/workflow` - Run workflow
- `/model` - Model info
- `/config` - Configuration
- `/stats` - Statistics
- `/tokens` - Token usage
- `/cost` - Cost tracking

See `docs/guides/TELEGRAM_BOT_COMMANDS.md` for details.
