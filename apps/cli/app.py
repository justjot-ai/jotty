"""
Jotty CLI - Terminal Interface
================================

Interactive terminal interface using shared UI components.

Usage:
    python -m apps.cli
    python -m Jotty.cli
"""

import asyncio
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.shared import ChatInterface, ChatState
from apps.shared.events import EventProcessor
from apps.shared.models import Message
from apps.shared.renderers import (
    TerminalInputHandler,
    TerminalMessageRenderer,
    TerminalStatusRenderer,
)
from Jotty.sdk import Jotty

# Check for prompt_toolkit
try:
    import prompt_toolkit

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class JottyCLI:
    """
    Jotty CLI - Interactive Terminal Interface

    Features:
    - Shared components across all platforms
    - Command registry with all 36 commands
    - REPL with history and autocomplete
    - Session management
    - Streaming responses with live updates
    - Intelligent routing (DIRECT/AUDIT_ONLY/FULL modes)
    """

    def __init__(
        self, config_path: Optional[str] = None, no_color: bool = False, debug: bool = False
    ):
        """Initialize CLI with shared components."""
        # Import CLI components
        from .commands import register_all_commands
        from .commands.base import CommandRegistry
        from .config.loader import ConfigLoader
        from .repl.engine import REPLEngine, SimpleREPL
        from .repl.session import SessionManager

        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.config.no_color = no_color
        self.config.debug = debug

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            # Hide internal Jotty logs from user - only show errors
            logging.basicConfig(level=logging.ERROR)
            # Suppress specific noisy loggers
            logging.getLogger("Jotty").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("litellm").setLevel(logging.ERROR)

        # Create chat interface
        self.chat = ChatInterface(
            message_renderer=TerminalMessageRenderer(),
            status_renderer=TerminalStatusRenderer(),
            input_handler=TerminalInputHandler(),
        )

        # Event processor
        self.event_processor = EventProcessor(self.chat)

        # Initialize global LLM provider (shared across all components)
        from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm

        try:
            self.lm = get_global_lm(provider="anthropic")
            logger.info(f"✅ Global LLM provider ready: {getattr(self.lm, 'model', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️  Global LLM provider init failed: {e}")

        # SDK client
        self.sdk = Jotty().use_local()

        # Command registry - ALL 36 commands
        self.command_registry = CommandRegistry()
        register_all_commands(self.command_registry)

        # Session manager
        self.session = SessionManager(
            session_dir=self.config.session.session_dir,
            context_window=self.config.session.context_window,
            auto_save=self.config.session.auto_save,
        )

        # REPL engine with history and autocomplete
        if PROMPT_TOOLKIT_AVAILABLE:
            self.repl = REPLEngine(
                command_registry=self.command_registry,
                history_file=self.config.session.history_file,
                prompt_text="jotty> ",
            )
        else:
            self.repl = SimpleREPL(
                command_registry=self.command_registry,
                prompt_text="jotty> ",
            )

        # Output history for /export command
        self._output_history = []

        logger.info("Jotty CLI initialized with shared components and all 36 commands")

    async def run(self):
        """Main REPL loop."""
        # Show welcome message
        welcome = Message(
            role="system",
            content="# Welcome to Jotty AI\n\nType your message or use /help for commands.",
        )
        self.chat.add_message(welcome)

        # Main loop
        while True:
            try:
                # Get user input
                user_input = await self.chat.input_handler.get_input("jotty> ")

                if not user_input:
                    break

                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Add user message
                user_msg = Message(role="user", content=user_input)
                self.chat.add_message(user_msg)

                # Execute via SDK with streaming
                async for event in self.sdk.stream(user_input):
                    # Process event (auto-updates UI)
                    await self.event_processor.process_event(event)

                # Clear ephemeral messages (typing indicators, etc.)
                self.chat.session.clear_ephemeral()

            except KeyboardInterrupt:
                if await self.chat.input_handler.confirm("Really exit?"):
                    break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                from apps.shared.models import Error

                error = Error(
                    message=str(e),
                    error_type=type(e).__name__,
                )
                self.chat.show_error(error)

    async def _handle_command(self, command: str):
        """Handle slash commands."""
        cmd = command.split()[0]

        if cmd == "/help":
            help_text = """
# Jotty CLI Commands

**Chat:**
- Just type your message to chat

**Commands:**
- /help - Show this help
- /clear - Clear chat history
- /status - Show status
- /swarm <agents> - Run swarm coordination
- /skill <name> - Execute skill
- /memory - Show memory status

**Exit:**
- Ctrl+C or Ctrl+D to exit
"""
            msg = Message(role="system", content=help_text)
            self.chat.add_message(msg)

        elif cmd == "/clear":
            self.chat.clear()
            msg = Message(role="system", content="Chat cleared!")
            self.chat.add_message(msg)

        elif cmd == "/status":
            state = self.chat.state_machine.get_state()
            msg = Message(
                role="system",
                content=f"Status: {state.value}\nMessages: {len(self.chat.session.messages)}",
            )
            self.chat.add_message(msg)

        elif cmd == "/swarm":
            # Extract agents from command
            parts = command.split(maxsplit=1)
            agents = parts[1] if len(parts) > 1 else "researcher,coder,tester"

            # Set state
            self.chat.set_state(ChatState.COORDINATING_SWARM)

            # Execute swarm via SDK
            async for event in self.sdk.swarm_stream(agents=agents, goal=agents):
                await self.event_processor.process_event(event)

        elif cmd == "/memory":
            # Get memory status via SDK
            status = await self.sdk.memory_status()
            msg = Message(
                role="system",
                content=f"Memory Status:\n```json\n{status}\n```",
                format="markdown",
            )
            self.chat.add_message(msg)

        else:
            msg = Message(role="system", content=f"Unknown command: {cmd}")
            self.chat.add_message(msg)


async def main():
    """Entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run CLI
    cli = JottyCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
