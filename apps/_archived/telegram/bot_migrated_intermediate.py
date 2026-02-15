"""
Telegram Bot (Migrated to Shared Components)
=============================================

Telegram bot using shared UI components for consistent rendering.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Optional

# CRITICAL FIX: Remove apps/ from sys.path to avoid shadowing telegram package
# When running as -m apps.telegram.bot_migrated, Python adds apps/ to sys.path
# which causes apps/telegram/ to shadow the real telegram package
apps_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if apps_path in sys.path:
    sys.path.remove(apps_path)

# Now we can safely import telegram
from dotenv import load_dotenv
from telegram.ext import Application, MessageHandler, filters

# Add Jotty root to path for imports
jotty_root = os.path.dirname(apps_path)
if jotty_root not in sys.path:
    sys.path.append(jotty_root)

from apps.shared import ChatInterface
from apps.shared.events import EventProcessor
from apps.shared.models import ChatSession, Message
from apps.shared.renderers import TelegramMessageRenderer, TelegramStatusRenderer
from Jotty.sdk import Jotty

logger = logging.getLogger(__name__)
load_dotenv()


class TelegramBotMigrated:
    """
    Telegram bot using shared components.

    **Before (OLD):**
    - Custom Telegram formatting logic
    - Manual message splitting
    - Duplicate event handling

    **After (NEW):**
    - Shared TelegramMessageRenderer
    - Auto message splitting
    - Shared EventProcessor
    """

    def __init__(self, token: Optional[str] = None):
        """Initialize Telegram bot."""
        self.token = token or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN not found in environment")

        # Session registry (chat_id -> ChatInterface)
        self.sessions: Dict[int, ChatInterface] = {}

        # SDK client
        self.sdk = Jotty()

        # Bot application (lazy init)
        self._application = None

        logger.info("Telegram bot initialized with shared components")

    def _get_or_create_session(self, chat_id: int, send_callback) -> ChatInterface:
        """Get or create chat interface for this chat."""
        if chat_id not in self.sessions:
            self.sessions[chat_id] = ChatInterface(
                message_renderer=TelegramMessageRenderer(send_callback),
                status_renderer=TelegramStatusRenderer(send_callback),
                input_handler=None,  # Not needed for bot
            )
        return self.sessions[chat_id]

    async def handle_message(self, update, context):
        """Handle incoming message."""
        chat_id = update.effective_chat.id
        message_text = update.message.text

        # Create send callback for this chat
        async def send_telegram_message(text: str):
            """Send message to Telegram with MarkdownV2 formatting."""
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="MarkdownV2",
            )

        # Get chat interface
        chat = self._get_or_create_session(chat_id, send_telegram_message)
        event_processor = EventProcessor(chat)

        # Add user message
        user_msg = Message(role="user", content=message_text)
        chat.add_message(user_msg)

        # Handle commands
        if message_text.startswith("/"):
            await self._handle_command(message_text, chat, update, context)
            return

        # Process via SDK with streaming
        try:
            async for event in self.sdk.chat_stream(
                message_text,
                session_id=str(chat_id),
            ):
                # Process event (auto-sends to Telegram)
                await event_processor.process_event(event)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            from apps.shared.models import Error

            error = Error(
                message=str(e),
                error_type=type(e).__name__,
                recoverable=True,
            )
            # Use status renderer to show error
            chat.status_renderer.render_error(error)

    async def _handle_command(self, command: str, chat: ChatInterface, update, context):
        """Handle Telegram slash commands."""
        cmd = command.split()[0]

        if cmd == "/start":
            welcome = """
*Welcome to Jotty AI Bot\\!* 🤖

I'm an AI assistant powered by Jotty\\.

*Commands:*
/help \\- Show this help
/status \\- Show bot status
/clear \\- Clear conversation
/swarm \\- Run swarm coordination

*Just send any message to chat\\!*

Examples:
• Search for AI news
• Write a Python script
• Explain quantum physics
"""
            await update.message.reply_text(welcome, parse_mode="MarkdownV2")

        elif cmd == "/help":
            help_text = """
*Jotty Telegram Bot Commands*

*Basic:*
/help \\- Show this help
/status \\- Show status
/clear \\- Clear chat

*Advanced:*
/swarm <agents> \\- Run swarm
/memory \\- Memory status
/skill <name> \\- Execute skill

*Chat:*
Just send any message\\!
"""
            await update.message.reply_text(help_text, parse_mode="MarkdownV2")

        elif cmd == "/status":
            state = chat.state_machine.get_state()
            status_text = f"""
📊 *Bot Status*

State: {state.value}
Messages: {len(chat.session.messages)}
Session: {chat.session.session_id}
"""
            await update.message.reply_text(status_text, parse_mode="MarkdownV2")

        elif cmd == "/clear":
            chat.clear()
            await update.message.reply_text("🗑️ *Chat cleared\\!*", parse_mode="MarkdownV2")

        elif cmd == "/swarm":
            # Extract agents from command
            parts = command.split(maxsplit=1)
            agents = parts[1] if len(parts) > 1 else "researcher,coder,tester"

            # Set state
            from apps.shared.state import ChatState

            chat.set_state(ChatState.COORDINATING_SWARM)

            # Execute swarm via SDK
            async for event in self.sdk.swarm_stream(agents=agents, goal=agents):
                event_processor = EventProcessor(chat)
                await event_processor.process_event(event)

        else:
            await update.message.reply_text(f"Unknown command: {cmd}")

    async def start(self):
        """Start the bot."""
        # Create application
        self._application = Application.builder().token(self.token).build()

        # Add message handler
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Add command handler
        self._application.add_handler(MessageHandler(filters.COMMAND, self.handle_message))

        # Initialize and start polling
        logger.info("Starting Telegram bot...")
        async with self._application:
            await self._application.initialize()
            await self._application.start()
            await self._application.updater.start_polling()
            logger.info("Bot is running! Press Ctrl+C to stop.")

            # Keep running until interrupted
            import signal

            stop_event = asyncio.Event()

            def signal_handler(sig, frame):
                logger.info("Received interrupt signal, stopping...")
                stop_event.set()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            await stop_event.wait()

            logger.info("Stopping bot...")
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()


async def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    bot = TelegramBotMigrated()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
