"""
Telegram Bot Handler
====================

Main Telegram bot logic for Jotty.
Processes messages through LeanExecutor and maintains sessions.
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TelegramBotHandler:
    """
    Telegram bot handler for Jotty.

    Uses python-telegram-bot library for Telegram integration.
    Processes messages through the shared LeanExecutor backend.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_chat_ids: Optional[list] = None,
        status_callback: Optional[Callable] = None
    ):
        """
        Initialize Telegram bot handler.

        Args:
            token: Telegram bot token (from .env if not provided)
            allowed_chat_ids: List of allowed chat IDs (None = allow all)
            status_callback: Callback for status updates
        """
        load_dotenv()

        self.token = token or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN not found in environment")

        # Parse allowed chat IDs
        self.allowed_chat_ids = allowed_chat_ids
        if self.allowed_chat_ids is None:
            chat_id_str = os.getenv("TELEGRAM_CHAT_ID")
            if chat_id_str:
                self.allowed_chat_ids = [int(x.strip()) for x in chat_id_str.split(",")]

        self.status_callback = status_callback

        # Lazy imports
        self._application = None
        self._executor = None
        self._registry = None
        self._lm_configured = False

    def _ensure_lm_configured(self):
        """Ensure DSPy LM is configured (same as CLI/Web)."""
        if self._lm_configured:
            return True

        import dspy

        # Check if already configured
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
            self._lm_configured = True
            return True

        try:
            # Use the same unified_lm_provider as CLI/Web
            from ..core.foundation.unified_lm_provider import configure_dspy_lm

            # Auto-detect: tries API providers first, then CLI providers
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                model_name = getattr(lm, 'model', None) or getattr(lm, 'model_name', 'unknown')
                logger.info(f"LLM configured for Telegram: {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")

        return False

    def _get_executor(self):
        """Get or create LeanExecutor instance."""
        # Ensure LM is configured before creating executor
        self._ensure_lm_configured()

        if self._executor is None:
            from ..core.orchestration.v2.lean_executor import LeanExecutor
            self._executor = LeanExecutor(
                status_callback=self._handle_status
            )
        return self._executor

    def _get_session_registry(self):
        """Get session registry."""
        if self._registry is None:
            from ..cli.repl.session import get_session_registry, InterfaceType
            self._registry = get_session_registry()
        return self._registry

    def _handle_status(self, stage: str, detail: str = ""):
        """Handle status updates from executor."""
        if self.status_callback:
            self.status_callback(stage, detail)
        logger.info(f"Status: {stage} - {detail}")

    def _check_allowed(self, chat_id: int) -> bool:
        """Check if chat ID is allowed."""
        if self.allowed_chat_ids is None:
            return True
        return chat_id in self.allowed_chat_ids

    async def _send_typing(self, chat_id: int, application):
        """Send typing indicator."""
        try:
            await application.bot.send_chat_action(
                chat_id=chat_id,
                action="typing"
            )
        except Exception as e:
            logger.debug(f"Failed to send typing: {e}")

    async def _handle_start(self, update, context):
        """Handle /start command."""
        from .renderer import TelegramRenderer

        chat_id = update.effective_chat.id

        if not self._check_allowed(chat_id):
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        await update.message.reply_text(
            f"👋 Welcome to Jotty Bot\\!\n\n"
            f"Your session ID: `tg_{chat_id}`\n\n"
            f"Send any message to get started, or use /help for commands\\.",
            parse_mode="MarkdownV2"
        )

    async def _handle_help(self, update, context):
        """Handle /help command."""
        from .renderer import TelegramRenderer

        if not self._check_allowed(update.effective_chat.id):
            return

        await update.message.reply_text(
            TelegramRenderer.format_help(),
            parse_mode="MarkdownV2"
        )

    async def _handle_status(self, update, context):
        """Handle /status command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        # Get session info
        from ..cli.repl.session import InterfaceType
        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            from .renderer import TelegramRenderer
            info = session.to_dict()
            await update.message.reply_text(
                TelegramRenderer.format_session_info(info),
                parse_mode="MarkdownV2"
            )
        else:
            await update.message.reply_text("✅ Bot is running\\. No active session\\.", parse_mode="MarkdownV2")

    async def _handle_history(self, update, context):
        """Handle /history command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        from ..cli.repl.session import InterfaceType
        from .renderer import TelegramRenderer

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            history = session.get_history(limit=10)
            await update.message.reply_text(
                TelegramRenderer.format_history(history),
                parse_mode="MarkdownV2"
            )
        else:
            await update.message.reply_text("📭 No conversation history\\.", parse_mode="MarkdownV2")

    async def _handle_clear(self, update, context):
        """Handle /clear command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        from ..cli.repl.session import InterfaceType

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            session.clear_history()
            session.save()
            await update.message.reply_text("🗑️ Conversation history cleared\\.", parse_mode="MarkdownV2")
        else:
            await update.message.reply_text("No session to clear\\.", parse_mode="MarkdownV2")

    async def _handle_session(self, update, context):
        """Handle /session command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        from ..cli.repl.session import InterfaceType
        from .renderer import TelegramRenderer

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=True, interface=InterfaceType.TELEGRAM)

        info = session.to_dict()
        await update.message.reply_text(
            TelegramRenderer.format_session_info(info),
            parse_mode="MarkdownV2"
        )

    async def _handle_message(self, update, context):
        """Handle incoming text messages."""
        from .renderer import TelegramRenderer
        from ..cli.repl.session import InterfaceType

        chat_id = update.effective_chat.id

        if not self._check_allowed(chat_id):
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        text = update.message.text or update.message.caption or ""

        if not text.strip():
            return

        # Send typing indicator
        await self._send_typing(chat_id, context.application)

        # Get or create session
        registry = self._get_session_registry()
        session_id = f"tg_{chat_id}"
        session = registry.get_session(session_id, create=True, interface=InterfaceType.TELEGRAM)

        # Add user message to session
        user_id = str(update.message.from_user.id) if update.message.from_user else str(chat_id)
        session.add_message(
            role="user",
            content=text,
            interface=InterfaceType.TELEGRAM,
            user_id=user_id,
            metadata={
                "message_id": update.message.message_id,
                "username": update.message.from_user.username if update.message.from_user else None,
            }
        )

        try:
            # Process through LeanExecutor
            executor = self._get_executor()

            # Send initial status
            status_msg = await update.message.reply_text(
                TelegramRenderer.format_status("Processing", "analyzing request..."),
            )

            # Execute task
            result = await executor.execute(text)

            # Delete status message
            try:
                await status_msg.delete()
            except Exception:
                pass

            if result.success:
                # Render and send response
                messages = TelegramRenderer.render(result.content, result.output_format)

                for i, msg in enumerate(messages):
                    try:
                        await update.message.reply_text(
                            msg,
                            parse_mode="MarkdownV2" if "```" not in msg else None
                        )
                    except Exception as e:
                        # Fallback without markdown
                        logger.warning(f"MarkdownV2 failed, retrying plain: {e}")
                        await update.message.reply_text(msg)

                    # Delay between messages
                    if i < len(messages) - 1:
                        await asyncio.sleep(0.5)

                # Add assistant response to session
                session.add_message(
                    role="assistant",
                    content=result.content,
                    interface=InterfaceType.TELEGRAM,
                    metadata={
                        "output_format": result.output_format,
                        "output_path": result.output_path,
                        "steps": result.steps_taken,
                    }
                )

                # Notify about saved file
                if result.output_path:
                    await update.message.reply_text(
                        f"📁 Output saved to: `{result.output_path}`",
                        parse_mode="MarkdownV2"
                    )

            else:
                await update.message.reply_text(
                    TelegramRenderer.format_error(result.error or "Unknown error")
                )

        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
            await update.message.reply_text(
                TelegramRenderer.format_error(str(e))
            )

    def setup_handlers(self, application):
        """Setup command and message handlers."""
        from telegram.ext import CommandHandler, MessageHandler, filters

        # Command handlers
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("help", self._handle_help))
        application.add_handler(CommandHandler("status", self._handle_status))
        application.add_handler(CommandHandler("history", self._handle_history))
        application.add_handler(CommandHandler("clear", self._handle_clear))
        application.add_handler(CommandHandler("session", self._handle_session))

        # Message handler (must be last)
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_message
            )
        )

    def run(self):
        """Run the bot (blocking)."""
        from telegram.ext import Application

        logger.info("Starting Telegram bot...")

        # Build application
        application = Application.builder().token(self.token).build()

        # Setup handlers
        self.setup_handlers(application)

        # Store for access in handlers
        self._application = application

        logger.info("Bot is running. Press Ctrl+C to stop.")

        # Run polling
        application.run_polling(
            allowed_updates=["message", "edited_message"],
            drop_pending_updates=True
        )

    async def run_async(self):
        """Run the bot asynchronously."""
        from telegram.ext import Application

        logger.info("Starting Telegram bot (async)...")

        # Build application
        application = Application.builder().token(self.token).build()

        # Setup handlers
        self.setup_handlers(application)

        self._application = application

        # Initialize and start
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            allowed_updates=["message", "edited_message"],
            drop_pending_updates=True
        )

        logger.info("Bot is running asynchronously.")

        return application

    async def stop(self):
        """Stop the bot."""
        if self._application:
            logger.info("Stopping Telegram bot...")
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            self._application = None
            logger.info("Bot stopped.")
