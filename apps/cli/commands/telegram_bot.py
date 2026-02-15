"""
Telegram Bot Command
====================

CLI command to start/manage the Telegram bot.

Usage:
    /telegram start    - Start the Telegram bot
    /telegram stop     - Stop the Telegram bot
    /telegram status   - Check bot status
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Any

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class TelegramCommand(BaseCommand):
    """Start and manage Telegram bot integration."""

    name = "telegram"
    aliases = ["tg", "bot"]
    description = "Start and manage Telegram bot"
    usage = "/telegram [start|stop|status]"
    category = "integrations"

    def __init__(self) -> None:
        self._bot = None
        self._task: Optional[asyncio.Task] = None

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute telegram command."""
        subcommand = args.positional[0] if args.positional else "start"

        if subcommand == "start":
            return await self._start_bot(cli)
        elif subcommand == "stop":
            return await self._stop_bot(cli)
        elif subcommand == "status":
            return self._get_status()
        elif subcommand == "help":
            return CommandResult.ok(self._get_help())
        else:
            return CommandResult.fail(f"Unknown subcommand: {subcommand}")

    async def _start_bot(self, cli: "JottyCLI") -> CommandResult:
        """Start the Telegram bot."""
        if self._bot is not None:
            return CommandResult.fail("Bot is already running. Use /telegram stop first.")

        try:
            from ...telegram.bot import TelegramBotHandler

            # Create status callback
            def status_callback(stage: str, detail: str = '') -> Any:
                if cli.renderer:
                    cli.renderer.status(f"[Telegram] {stage}: {detail}")

            self._bot = TelegramBotHandler(status_callback=status_callback)

            # Start bot in background task
            async def run_bot() -> Any:
                try:
                    await self._bot.run_async()
                    # Keep running until stopped
                    while self._bot._application:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Bot task cancelled")
                except Exception as e:
                    logger.error(f"Bot error: {e}")
                finally:
                    if self._bot:
                        await self._bot.stop()

            self._task = asyncio.create_task(run_bot())

            return CommandResult.ok(
                "Telegram bot started successfully!\n"
                "The bot is now listening for messages.\n"
                "Use /telegram stop to stop the bot."
            )

        except ImportError as e:
            return CommandResult.fail(
                f"Missing dependency: {e}\n"
                "Install with: pip install python-telegram-bot"
            )
        except ValueError as e:
            return CommandResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to start bot: {e}", exc_info=True)
            return CommandResult.fail(f"Failed to start bot: {e}")

    async def _stop_bot(self, cli: "JottyCLI") -> CommandResult:
        """Stop the Telegram bot."""
        if self._bot is None:
            return CommandResult.fail("Bot is not running.")

        try:
            # Cancel the task
            if self._task:
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._task = None

            # Stop the bot
            if self._bot:
                await self._bot.stop()
                self._bot = None

            return CommandResult.ok("Telegram bot stopped.")

        except Exception as e:
            logger.error(f"Error stopping bot: {e}", exc_info=True)
            self._bot = None
            self._task = None
            return CommandResult.fail(f"Error stopping bot: {e}")

    def _get_status(self) -> CommandResult:
        """Get bot status."""
        if self._bot is None:
            return CommandResult.ok(
                "Telegram Bot Status: STOPPED\n"
                "Use /telegram start to start the bot."
            )

        running = self._bot._application is not None

        status_lines = [
            f"Telegram Bot Status: {'RUNNING' if running else 'STARTING'}",
            "",
        ]

        if self._bot.allowed_chat_ids:
            status_lines.append(f"Allowed chat IDs: {', '.join(map(str, self._bot.allowed_chat_ids))}")
        else:
            status_lines.append("Allowed chat IDs: All")

        return CommandResult.ok("\n".join(status_lines))

    def _get_help(self) -> str:
        """Get help text."""
        return """Telegram Bot Command

Start and manage the Telegram bot integration.

Usage:
    /telegram start   - Start the Telegram bot (runs in background)
    /telegram stop    - Stop the Telegram bot
    /telegram status  - Check if bot is running

Configuration:
    Set these in your .env file:
    - TELEGRAM_TOKEN: Your bot token from @BotFather
    - TELEGRAM_CHAT_ID: Allowed chat ID(s), comma-separated

The bot shares sessions with CLI, so conversations are synced.
"""

    def get_completions(self, partial: str) -> list:
        """Get completions for subcommands."""
        subcommands = ["start", "stop", "status", "help"]
        return [s for s in subcommands if s.startswith(partial)]
