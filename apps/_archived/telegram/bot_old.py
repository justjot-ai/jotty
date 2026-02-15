"""
Telegram Bot Handler
====================

Main Telegram bot logic for Jotty.
Processes messages through ChatExecutor and maintains sessions.
Supports CLI slash commands via shared CommandRegistry.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class TelegramBotHandler:
    """
    Telegram bot handler for Jotty.

    Uses python-telegram-bot library for Telegram integration.
    Processes messages through the shared ChatExecutor backend.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_chat_ids: Optional[list] = None,
        status_callback: Optional[Callable] = None,
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
        self._session_registry = None
        self._lm_configured = False
        self._command_registry = None
        self._skills_registry = None

    def _ensure_lm_configured(self):
        """Ensure DSPy LM is configured (same as CLI)."""
        if self._lm_configured:
            logger.debug("LM already configured (cached)")
            return True

        import dspy

        # Check if already configured
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            self._lm_configured = True
            logger.info(f"LM already configured in dspy.settings: {dspy.settings.lm}")
            return True

        logger.info("Configuring LLM for Telegram bot...")

        # Suppress warnings during model loading (same as CLI)
        import warnings

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")

        try:
            from core.foundation.unified_lm_provider import configure_dspy_lm

            # Auto-detect: tries API providers first, then CLI providers
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                model_name = getattr(lm, "model", None) or getattr(lm, "model_name", "unknown")
                if "/" in str(model_name):
                    provider_name, model_short = str(model_name).split("/", 1)
                else:
                    provider_name = type(lm).__name__
                    model_short = str(model_name)
                logger.info(f"LLM configured: {provider_name} ({model_short[:30]})")
                return True
            else:
                logger.error("configure_dspy_lm() returned None")
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}", exc_info=True)

        return False

    def _get_command_registry(self):
        """Get CLI command registry for slash command support."""
        if self._command_registry is None:
            try:
                # Use unified CommandService (same as CLI, Web, Supervisor)
                from core.services.command_service import get_command_service

                service = get_command_service()
                service._ensure_initialized()
                self._command_registry = service._registry
                logger.info(
                    f"Command registry initialized via unified CommandService with {len(self._command_registry._commands)} commands"
                )
            except Exception as e:
                logger.error(f"Failed to initialize command registry: {e}")
                # Fallback to direct import
                try:
                    from Jotty.apps.cli.commands import register_all_commands
                    from Jotty.apps.cli.commands.base import CommandRegistry

                    self._command_registry = CommandRegistry()
                    register_all_commands(self._command_registry)
                    logger.info(
                        f"Command registry initialized directly with {len(self._command_registry._commands)} commands"
                    )
                except Exception as e2:
                    logger.error(f"Fallback command registry also failed: {e2}")
        return self._command_registry

    def _get_skills_registry(self):
        """Get skills registry."""
        if self._skills_registry is None:
            try:
                from core.registry.skills_registry import get_skills_registry

                self._skills_registry = get_skills_registry()
                if not self._skills_registry.initialized:
                    self._skills_registry.init()
                logger.info(
                    f"Skills registry initialized with {len(self._skills_registry.loaded_skills)} skills"
                )
            except Exception as e:
                logger.error(f"Failed to initialize skills registry: {e}")
        return self._skills_registry

    def _get_executor(self):
        """Get or create ChatExecutor instance (auto-detects provider)."""
        if self._executor is None:
            from core.orchestration.unified_executor import ChatExecutor

            self._executor = ChatExecutor(status_callback=self._handle_status)
        return self._executor

    def _get_session_registry(self):
        """Get session registry."""
        if self._session_registry is None:
            get_session_registry, _ = self._import_session_module()
            self._session_registry = get_session_registry()
        return self._session_registry

    def _import_session_module(self):
        """Import session module with fallback for different run modes."""
        try:
            from ..cli.repl.session import InterfaceType, get_session_registry
        except ImportError:
            from Jotty.apps.cli.repl.session import InterfaceType, get_session_registry
        return get_session_registry, InterfaceType

    def _get_interface_type(self):
        """Get InterfaceType enum."""
        _, InterfaceType = self._import_session_module()
        return InterfaceType

    def _handle_status(self, stage: str, detail: str = ""):
        """Handle status updates from executor."""
        if self.status_callback:
            self.status_callback(stage, detail)
        logger.info(f"Status: {stage} - {detail}")

    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown to Telegram HTML format."""
        import html
        import re

        lines = text.split("\n")
        result = []

        for line in lines:
            # Skip escaping for decoration lines
            if line.strip() in ["═" * 30, "─" * 30, "━━━", "---", "***"]:
                result.append(line)
                continue

            # Escape HTML entities first
            escaped = html.escape(line)

            # Convert markdown to HTML
            # Bold: **text** or __text__ -> <b>text</b>
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
            escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)

            # Italic: *text* -> <i>text</i> (but not if part of **)
            escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)

            # Code: `text` -> <code>text</code>
            escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)

            # Headers: ## text -> <b>text</b>
            escaped = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", escaped)

            # Bullet points: - text or * text -> • text
            escaped = re.sub(r"^[\-\*]\s+", "• ", escaped)

            result.append(escaped)

        return "\n".join(result)

    def _check_allowed(self, chat_id: int) -> bool:
        """Check if chat ID is allowed."""
        if self.allowed_chat_ids is None:
            return True
        return chat_id in self.allowed_chat_ids

    async def _send_file_to_telegram(self, update, file_path: str):
        """
        Send a generated file back through Telegram.

        This ensures outputs are delivered through the same channel
        the user is using (Telegram, CLI, Web, etc.)
        """
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            await update.message.reply_text(f"📁 File created: {file_path}")
            return

        try:
            suffix = path.suffix.lower()

            # Send as document with appropriate caption
            caption = f"📄 {path.name}"

            if suffix == ".pdf":
                await update.message.reply_document(
                    document=open(path, "rb"), filename=path.name, caption=caption
                )
            elif suffix in [".docx", ".doc"]:
                await update.message.reply_document(
                    document=open(path, "rb"), filename=path.name, caption=caption
                )
            elif suffix in [".png", ".jpg", ".jpeg", ".gif"]:
                await update.message.reply_photo(photo=open(path, "rb"), caption=caption)
            elif suffix in [".md", ".txt", ".json", ".csv"]:
                # For text files, send as document
                await update.message.reply_document(
                    document=open(path, "rb"), filename=path.name, caption=caption
                )
            else:
                # Generic file
                await update.message.reply_document(
                    document=open(path, "rb"), filename=path.name, caption=caption
                )

            logger.info(f"Sent file to Telegram: {file_path}")

        except Exception as e:
            logger.error(f"Failed to send file to Telegram: {e}")
            await update.message.reply_text(
                f"📁 File created: {file_path}\n(Could not send directly: {e})"
            )

    async def _send_typing(self, chat_id: int, application):
        """Send typing indicator."""
        try:
            await application.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception as e:
            logger.debug(f"Failed to send typing: {e}")

    async def _handle_start(self, update, context):
        """Handle /start command."""
        from .renderer import TelegramRenderer

        chat_id = update.effective_chat.id

        if not self._check_allowed(chat_id):
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return

        await update.message.reply_text(
            f"👋 Welcome to Jotty Bot\\!\n\n"
            f"Your session ID: `tg_{chat_id}`\n\n"
            f"Send any message to get started, or use /help for commands\\.",
            parse_mode="MarkdownV2",
        )

    async def _handle_help(self, update, context):
        """Handle /help command."""
        from .renderer import TelegramRenderer

        if not self._check_allowed(update.effective_chat.id):
            return

        await update.message.reply_text(TelegramRenderer.format_help(), parse_mode="MarkdownV2")

    async def _handle_status(self, update, context):
        """Handle /status command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        # Get session info
        InterfaceType = self._get_interface_type()
        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            from .renderer import TelegramRenderer

            info = session.to_dict()
            await update.message.reply_text(
                TelegramRenderer.format_session_info(info), parse_mode="MarkdownV2"
            )
        else:
            await update.message.reply_text(
                "✅ Bot is running\\. No active session\\.", parse_mode="MarkdownV2"
            )

    async def _handle_history(self, update, context):
        """Handle /history command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        InterfaceType = self._get_interface_type()
        from .renderer import TelegramRenderer

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            history = session.get_history(limit=10)
            await update.message.reply_text(
                TelegramRenderer.format_history(history), parse_mode="MarkdownV2"
            )
        else:
            await update.message.reply_text(
                "📭 No conversation history\\.", parse_mode="MarkdownV2"
            )

    async def _handle_clear(self, update, context):
        """Handle /clear command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        InterfaceType = self._get_interface_type()

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=False, interface=InterfaceType.TELEGRAM)

        if session:
            session.clear_history()
            session.save()
            await update.message.reply_text(
                "🗑️ Conversation history cleared\\.", parse_mode="MarkdownV2"
            )
        else:
            await update.message.reply_text("No session to clear\\.", parse_mode="MarkdownV2")

    async def _handle_session(self, update, context):
        """Handle /session command."""
        if not self._check_allowed(update.effective_chat.id):
            return

        InterfaceType = self._get_interface_type()
        from .renderer import TelegramRenderer

        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=True, interface=InterfaceType.TELEGRAM)

        info = session.to_dict()
        await update.message.reply_text(
            TelegramRenderer.format_session_info(info), parse_mode="MarkdownV2"
        )

    async def _handle_message(self, update, context):
        """Handle incoming text messages - routes to CLI commands or ChatExecutor."""
        from .renderer import TelegramRenderer

        InterfaceType = self._get_interface_type()

        chat_id = update.effective_chat.id

        if not self._check_allowed(chat_id):
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
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
            },
        )

        try:
            # Check if this is a CLI slash command (e.g., /run, /skills, /agents)
            if (
                text.startswith("/")
                and not text.startswith("/start")
                and not text.startswith("/help")
            ):
                result = await self._handle_cli_command(text, update, session, InterfaceType)
                return

            # Natural language - process through ChatExecutor with streaming
            from core.orchestration.unified_executor import ChatExecutor

            # Send initial message for streaming updates
            stream_msg = await update.message.reply_text("⏳ Thinking...")

            # Streaming state
            stream_content = []
            last_update_time = [0]
            UPDATE_INTERVAL = 0.8  # Update message every 0.8 seconds

            async def stream_callback(chunk: str):
                """Update Telegram message with streaming content."""
                stream_content.append(chunk)
                current_time = asyncio.get_running_loop().time()

                # Only update message periodically to avoid rate limits
                if current_time - last_update_time[0] >= UPDATE_INTERVAL:
                    last_update_time[0] = current_time
                    full_text = "".join(stream_content)
                    # Truncate if too long (Telegram limit)
                    display_text = full_text[-3500:] if len(full_text) > 3500 else full_text
                    if display_text:
                        try:
                            await stream_msg.edit_text(f"✍️ {display_text}...")
                        except Exception as e:
                            logger.debug(f"Stream edit failed: {e}")

            def status_callback(stage: str, detail: str = ""):
                """Status callback for executor."""
                logger.info(f"Status: {stage} - {detail}")

            # Create executor with streaming (auto-detects provider)
            executor = ChatExecutor(
                status_callback=status_callback,
                stream_callback=lambda chunk: asyncio.create_task(stream_callback(chunk)),
            )

            # Execute task
            result = await executor.execute(text)

            # Final update with complete response
            if result.success:
                # Get final content
                final_content = result.content

                # Edit message with final content (convert markdown to HTML)
                try:
                    # Truncate for Telegram (4096 char limit)
                    display_content = (
                        final_content[:4000] if len(final_content) > 4000 else final_content
                    )
                    # Convert markdown to HTML for proper rendering
                    html_content = self._markdown_to_html(display_content)
                    await stream_msg.edit_text(html_content, parse_mode="HTML")
                except Exception as e:
                    logger.warning(f"HTML edit failed: {e}, trying plain text")
                    # Fallback to plain text if HTML fails
                    try:
                        await stream_msg.edit_text(display_content)
                    except Exception as e2:
                        logger.warning(f"Plain text edit also failed: {e2}")
                        # Send as new message if edit fails
                        await stream_msg.delete()
                        messages = TelegramRenderer.render(final_content, result.output_format)
                        for msg in messages:
                            await update.message.reply_text(msg)

                # Add assistant response to session
                session.add_message(
                    role="assistant",
                    content=result.content,
                    interface=InterfaceType.TELEGRAM,
                    metadata={
                        "output_format": result.output_format,
                        "output_path": result.output_path,
                        "steps": result.steps_taken,
                    },
                )

                # Send file if generated (deliver through Telegram, not just path)
                if result.output_path:
                    await self._send_file_to_telegram(update, result.output_path)

            else:
                await update.message.reply_text(
                    TelegramRenderer.format_error(result.error or "Unknown error")
                )

        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
            await update.message.reply_text(TelegramRenderer.format_error(str(e)))

    async def _handle_cli_command(self, text: str, update, session, InterfaceType):
        """
        Handle CLI slash commands via shared CommandRegistry.

        Supports all CLI commands like /run, /skills, /agents, /swarm, etc.
        """
        from .renderer import TelegramRenderer

        # Parse command
        parts = text[1:].split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        # Get command registry
        cmd_registry = self._get_command_registry()
        if not cmd_registry:
            await update.message.reply_text("Command system not available.")
            return

        # Find command - try original name first, then with hyphen (stock_ml -> stock-ml)
        command = cmd_registry.get(cmd_name)
        if not command:
            # Try converting underscore to hyphen (Telegram uses underscore, CLI uses hyphen)
            alt_name = cmd_name.replace("_", "-")
            command = cmd_registry.get(alt_name)
        if not command:
            # List available commands
            available = ", ".join(sorted(cmd_registry._commands.keys())[:15])
            await update.message.reply_text(
                f"Unknown command: /{cmd_name}\n\nAvailable: {available}..."
            )
            return

        # Create a minimal CLI-like context for command execution
        class TelegramCLIContext:
            """Minimal CLI context for Telegram command execution."""

            def __init__(self, bot_handler, update_obj):
                self.bot_handler = bot_handler
                self.update = update_obj
                self.renderer = TelegramCLIRenderer(update_obj)
                self._swarm_manager = None
                self._output_history = []

                # Build config with all required attributes
                swarm_config = type(
                    "SwarmConfig",
                    (),
                    {
                        "enable_learning": False,  # Used by learn command
                        "learning_enabled": False,  # Alias
                        "mode": "single",
                        "max_agents": 1,
                    },
                )()

                session_config = type(
                    "SessionConfig", (), {"auto_save": False, "history_size": 100}
                )()

                self.config = type(
                    "Config",
                    (),
                    {
                        "debug": False,
                        "session": session_config,
                        "swarm": swarm_config,
                        "learning": type("LearningConfig", (), {"enabled": False})(),
                        "output_dir": "~/jotty/outputs",
                    },
                )()

            async def get_swarm_manager(self):
                """Get swarm manager (lazy)."""
                if self._swarm_manager is None:
                    from core.foundation.data_structures import SwarmConfig
                    from core.orchestration import Orchestrator

                    self._swarm_manager = Orchestrator(config=SwarmConfig())
                return self._swarm_manager

            def get_skills_registry(self):
                return self.bot_handler._get_skills_registry()

        class TelegramCLIRenderer:
            """Full CLI renderer adapter for Telegram with real-time status streaming."""

            def __init__(self, update_obj):
                self.update = update_obj
                self._buffer = []
                self._status_message = None  # For editing status in place
                self._status_history = []  # Track status updates
                self.tables = self.TablesAdapter(self)
                self.progress = self.ProgressAdapter(self)

            # Basic output methods
            def info(self, msg):
                self._buffer.append(f"ℹ️ {msg}")

            def success(self, msg):
                self._buffer.append(f"✅ {msg}")

            def warning(self, msg):
                self._buffer.append(f"⚠️ {msg}")

            def error(self, msg):
                self._buffer.append(f"❌ {msg}")

            def print(self, msg):
                self._buffer.append(str(msg))

            def newline(self):
                pass

            # Headers and structure
            def header(self, msg):
                self._buffer.append(f"\n{'═'*30}\n📌 {msg}\n{'═'*30}")

            def subheader(self, msg):
                self._buffer.append(f"\n▸ {msg}")

            def status(self, msg):
                """Buffer status - use send_status_async for real-time updates."""
                self._buffer.append(f"⏳ {msg}")
                self._status_history.append(msg)

            async def send_status_async(self, msg):
                """Send status update immediately to Telegram (edit existing or send new)."""
                import asyncio

                self._status_history.append(msg)

                # Build status display with history (plain text, no Markdown)
                status_text = "📊 Progress:\n"
                for i, s in enumerate(self._status_history[-5:]):  # Show last 5 statuses
                    if i == len(self._status_history[-5:]) - 1:
                        status_text += f"⏳ {s}\n"  # Current
                    else:
                        status_text += f"✅ {s}\n"  # Completed

                try:
                    if self._status_message:
                        # Edit existing status message (plain text)
                        await self._status_message.edit_text(status_text)
                    else:
                        # Send new status message (plain text, no parse_mode)
                        self._status_message = await self.update.message.reply_text(status_text)
                except Exception as e:
                    # If edit fails, try sending new message
                    try:
                        self._status_message = await self.update.message.reply_text(f"⏳ {msg}")
                    except:
                        logger.debug(f"Status update failed: {e}")

            async def clear_status_message(self):
                """Delete the status message after completion."""
                if self._status_message:
                    try:
                        await self._status_message.delete()
                    except:
                        pass
                    self._status_message = None

            def divider(self):
                self._buffer.append("─" * 30)

            def rule(self):
                self._buffer.append("─" * 30)

            def panel(self, content, **kwargs):
                title = kwargs.get("title", "")
                if title:
                    self._buffer.append(f"━━━ {title} ━━━\n{content}")
                else:
                    self._buffer.append(content)

            def markdown(self, content):
                self._buffer.append(content)

            # Search/results display
            def search_query(self, query, count=None):
                if count:
                    self._buffer.append(f"🔍 {query} ({count} results)")
                else:
                    self._buffer.append(f"🔍 {query}")

            def tool_output(self, tool_name, path=None):
                if path:
                    self._buffer.append(f"✅ {tool_name}: {path}")
                else:
                    self._buffer.append(f"✅ {tool_name}")

            def step_progress(self, step, total, desc, status="running"):
                icon = {"running": "⏳", "done": "✅", "failed": "❌"}.get(status, "▸")
                self._buffer.append(f"{icon} Step {step}/{total}: {desc}")

            # Additional methods used by CLI commands
            def result(self, content):
                self._buffer.append(f"📊 Result:\n{content}")

            def code(self, content, lang=""):
                self._buffer.append(f"```{lang}\n{content}\n```")

            def goodbye(self, msg=""):
                self._buffer.append(f"👋 {msg or 'Goodbye!'}")

            def clear(self):
                pass  # No-op for Telegram

            def prompt(self):
                return "jotty> "  # Not used in Telegram but might be called

            def tree(self, data, title=""):
                """Render dict/tree as formatted text."""
                lines = [f"📋 {title}" if title else ""]

                def _format_dict(d, indent=0):
                    for k, v in d.items():
                        prefix = "  " * indent
                        if isinstance(v, dict):
                            lines.append(f"{prefix}• {k}:")
                            _format_dict(v, indent + 1)
                        elif isinstance(v, list):
                            lines.append(f"{prefix}• {k}: {', '.join(str(x) for x in v[:5])}")
                            if len(v) > 5:
                                lines.append(f"{prefix}  ...and {len(v) - 5} more")
                        else:
                            lines.append(f"{prefix}• {k}: {v}")

                if isinstance(data, dict):
                    _format_dict(data)
                self._buffer.append("\n".join(lines))

            class TablesAdapter:
                """Adapter for table rendering."""

                def __init__(self, renderer):
                    self.renderer = renderer

                def skills_table(self, skills):
                    """Format skills as text table."""
                    lines = ["🔧 Skills:\n"]
                    for skill in skills[:20]:  # Limit to 20
                        name = skill.get("name", "unknown")
                        desc = skill.get("description", "")[:50]
                        tools_count = len(skill.get("tools", []))
                        lines.append(f"• {name} ({tools_count} tools)\n  {desc}")
                    if len(skills) > 20:
                        lines.append(f"\n...and {len(skills) - 20} more")
                    return "\n".join(lines)

                def print_table(self, table_text):
                    """Print the formatted table."""
                    self.renderer._buffer.append(table_text)

            class ProgressAdapter:
                """Adapter for progress indicators."""

                def __init__(self, renderer):
                    self.renderer = renderer

                def spinner(self, message=""):
                    """Return a context manager for spinner (no-op for Telegram)."""
                    from contextlib import contextmanager

                    @contextmanager
                    def _spinner():
                        self.renderer._buffer.append(f"⏳ {message}")
                        yield

                    return _spinner()

            async def flush(self):
                if self._buffer:
                    text = "\n".join(self._buffer)

                    # Split into chunks if too long (Telegram limit 4096)
                    chunks = self._split_message(text, 3800)

                    for chunk in chunks:
                        # Try HTML format first (more reliable than MarkdownV2)
                        try:
                            html_text = self._markdown_to_html(chunk)
                            await self.update.message.reply_text(html_text, parse_mode="HTML")
                        except Exception as e:
                            # Fallback to plain text
                            logger.debug(f"HTML send failed: {e}")
                            try:
                                await self.update.message.reply_text(chunk)
                            except Exception as e2:
                                # Last resort: truncate
                                try:
                                    await self.update.message.reply_text(
                                        chunk[:3500] + "\n...(truncated)"
                                    )
                                except Exception as e3:
                                    logger.error(f"Failed to send message: {e3}")

                    self._buffer = []

            def _markdown_to_html(self, text):
                """Convert basic markdown to Telegram HTML format."""
                import html
                import re

                # First escape HTML entities
                # But preserve our emoji markers
                lines = text.split("\n")
                result = []

                for line in lines:
                    # Skip escaping for lines with just emojis/markers
                    if line.strip() in ["═" * 30, "─" * 30, "━━━"]:
                        result.append(line)
                        continue

                    # Escape HTML but preserve structure
                    escaped = html.escape(line)

                    # Convert markdown to HTML
                    # Bold: **text** or __text__ -> <b>text</b>
                    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
                    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)

                    # Italic: *text* or _text_ -> <i>text</i>
                    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)

                    # Code: `text` -> <code>text</code>
                    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)

                    # Headers: ## text -> <b>text</b>
                    escaped = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", escaped)

                    result.append(escaped)

                return "\n".join(result)

            def _split_message(self, text, max_len=4000):
                """Split long message into chunks."""
                if len(text) <= max_len:
                    return [text]

                chunks = []
                while text:
                    if len(text) <= max_len:
                        chunks.append(text)
                        break
                    # Find a good break point
                    break_point = text.rfind("\n", 0, max_len)
                    if break_point == -1:
                        break_point = max_len
                    chunks.append(text[:break_point])
                    text = text[break_point:].lstrip()
                return chunks

            def _format_for_telegram(self, text):
                """Convert markdown to Telegram MarkdownV2 format."""
                import re

                # Escape special characters for MarkdownV2
                # Must escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
                def escape_md(s):
                    chars = r"_*[]()~`>#+=|{}.!-"
                    for c in chars:
                        s = s.replace(c, f"\\{c}")
                    return s

                # Process line by line to handle headers and formatting
                lines = text.split("\n")
                result = []

                for line in lines:
                    # Headers: ## Title -> *Title*
                    if line.startswith("### "):
                        result.append(f"*{escape_md(line[4:])}*")
                    elif line.startswith("## "):
                        result.append(f"*{escape_md(line[3:])}*")
                    elif line.startswith("# "):
                        result.append(f"*{escape_md(line[2:])}*")
                    # Bold: **text** -> *text*
                    elif "**" in line:
                        # Convert **bold** to *bold* and escape rest
                        parts = re.split(r"\*\*(.+?)\*\*", line)
                        processed = []
                        for i, part in enumerate(parts):
                            if i % 2 == 1:  # Bold part
                                processed.append(f"*{escape_md(part)}*")
                            else:
                                processed.append(escape_md(part))
                        result.append("".join(processed))
                    # Bullet points
                    elif line.startswith("- ") or line.startswith("• "):
                        result.append(f"• {escape_md(line[2:])}")
                    elif line.startswith("* "):
                        result.append(f"• {escape_md(line[2:])}")
                    # Code blocks (keep as is but escape)
                    elif line.startswith("```"):
                        result.append(line)  # Don't escape code block markers
                    # Links: [text](url) - keep format but escape text
                    elif re.match(r".*\[.+\]\(.+\).*", line):
                        # Handle links specially
                        def replace_link(m):
                            text_part = m.group(1)
                            url_part = m.group(2)
                            return f"[{escape_md(text_part)}]({url_part})"

                        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, line)
                        result.append(line)
                    else:
                        result.append(escape_md(line))

                return "\n".join(result)

        # Execute command with stdout capture (some CLI commands use print())
        try:
            import sys
            from io import StringIO

            ctx = TelegramCLIContext(self, update)
            args = command.parse_args(cmd_args)

            # Capture stdout since some commands use print() instead of renderer
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                result = await command.execute(args, ctx)
            finally:
                sys.stdout = old_stdout

            # Add captured stdout to buffer
            stdout_content = captured_output.getvalue()
            if stdout_content.strip():
                ctx.renderer._buffer.append(stdout_content.strip())

            # Send buffered output
            await ctx.renderer.flush()

            # Send result message if any
            if hasattr(result, "message") and result.message:
                messages = TelegramRenderer.render(result.message, "text")
                for msg in messages:
                    await update.message.reply_text(msg)

            # Send file if generated (deliver through Telegram channel)
            output_path = None
            if hasattr(result, "data") and isinstance(result.data, dict):
                output_path = result.data.get("output_path")
            if not output_path and hasattr(result, "output_path"):
                output_path = result.output_path

            if output_path:
                await self._send_file_to_telegram(update, output_path)

            # Add to session
            session.add_message(
                role="assistant",
                content=f"Command /{cmd_name} executed",
                interface=InterfaceType.TELEGRAM,
                metadata={
                    "command": cmd_name,
                    "success": result.success if hasattr(result, "success") else True,
                },
            )

        except Exception as e:
            logger.error(f"Command execution error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Command error: {e}")

    def setup_handlers(self, application):
        """Setup command and message handlers."""
        from telegram.ext import CommandHandler, MessageHandler, filters

        # Telegram-specific command handlers
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("help", self._handle_help))
        application.add_handler(CommandHandler("status", self._handle_status))
        application.add_handler(CommandHandler("history", self._handle_history))
        application.add_handler(CommandHandler("clear", self._handle_clear))
        application.add_handler(CommandHandler("session", self._handle_session))

        # CLI commands handler - routes /run, /skills, /agents, etc. to CLI
        application.add_handler(
            MessageHandler(
                filters.Regex(r"^/(?!start|help|status|history|clear|session)\w+"),
                self._handle_cli_command_wrapper,
            )
        )

        # Natural language message handler (must be last)
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    async def _handle_cli_command_wrapper(self, update, context):
        """Wrapper to handle CLI commands from Telegram."""
        if not self._check_allowed(update.effective_chat.id):
            await update.message.reply_text("Not authorized.")
            return

        text = update.message.text or ""
        InterfaceType = self._get_interface_type()
        registry = self._get_session_registry()
        session_id = f"tg_{update.effective_chat.id}"
        session = registry.get_session(session_id, create=True, interface=InterfaceType.TELEGRAM)

        await self._handle_cli_command(text, update, session, InterfaceType)

    async def _set_telegram_commands(self, application):
        """Set bot commands for Telegram command menu - synced from CLI registry."""
        from telegram import BotCommand

        # Start with Telegram-specific commands
        commands = [
            BotCommand("start", "Start the bot"),
            BotCommand("status", "Show bot status"),
            BotCommand("history", "Show conversation history"),
            BotCommand("clear", "Clear conversation history"),
            BotCommand("session", "Show session info"),
        ]

        # Add ALL CLI commands from registry (synced!)
        cmd_registry = self._get_command_registry()
        if cmd_registry:
            import re

            for name, cmd in sorted(cmd_registry._commands.items()):
                # Skip telegram command (we're already in telegram) and UI-only commands
                if name in ["telegram", "web", "browse", "preview", "resume"]:
                    continue

                # Sanitize command name for Telegram (lowercase, alphanumeric + underscore only, min 2 chars)
                tg_name = name.lower().replace("-", "_")
                # Skip if invalid (single char, too long, or bad chars)
                if len(tg_name) < 2 or not re.match(r"^[a-z][a-z0-9_]{1,31}$", tg_name):
                    logger.debug(f"Skipping invalid command name for Telegram: {name}")
                    continue

                # Get description, truncate to 256 chars (Telegram limit)
                desc = getattr(cmd, "description", f"{name} command")[:256]
                # Avoid duplicates
                if not any(c.command == tg_name for c in commands):
                    commands.append(BotCommand(tg_name, desc))

        # Telegram limits to 100 commands
        commands = commands[:100]

        try:
            await application.bot.set_my_commands(commands)
            logger.info(f"Telegram command menu synced with {len(commands)} commands from CLI")
        except Exception as e:
            logger.warning(f"Failed to set Telegram commands: {e}")

    def run(self):
        """Run the bot (blocking)."""
        from telegram.ext import Application

        logger.info("Starting Telegram bot...")

        # Configure LM at startup (same as CLI)
        if not self._ensure_lm_configured():
            logger.error("Failed to configure LLM - bot may not work properly")
        else:
            logger.info("LLM configured successfully at startup")

        # Initialize command registry for CLI commands
        cmd_registry = self._get_command_registry()
        if cmd_registry:
            logger.info(f"CLI commands available: {len(cmd_registry._commands)}")

        # Initialize skills registry
        skills = self._get_skills_registry()
        if skills:
            logger.info(f"Skills available: {len(skills.loaded_skills)}")

        # Build application with post_init to set commands
        async def post_init(app):
            await self._set_telegram_commands(app)

        application = Application.builder().token(self.token).post_init(post_init).build()

        # Setup handlers
        self.setup_handlers(application)

        # Store for access in handlers
        self._application = application

        logger.info("Bot is running. Press Ctrl+C to stop.")

        # Run polling
        application.run_polling(
            allowed_updates=["message", "edited_message"], drop_pending_updates=True
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
            allowed_updates=["message", "edited_message"], drop_pending_updates=True
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
