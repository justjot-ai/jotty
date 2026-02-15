"""
REPL Engine
===========

Main REPL (Read-Eval-Print Loop) engine using prompt_toolkit.
"""

import sys
import logging
from typing import TYPE_CHECKING, Optional, Callable, Awaitable, Any

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    PromptSession = None
    Style = None
    HTML = None

from .history import HistoryManager
from .completer import CommandCompleter, SimpleCompleter
from ..ui.renderer import FooterHints, REPLState

if TYPE_CHECKING:
    from ..commands.base import CommandRegistry

logger = logging.getLogger(__name__)


class REPLEngine:
    """
    REPL Engine for Jotty CLI.

    Features:
    - Rich prompt with prompt_toolkit
    - Command history with persistence
    - Autocomplete for commands and skills
    - Vi/Emacs key bindings
    """

    def __init__(self, command_registry: 'CommandRegistry', history_file: Optional[str] = None, prompt_text: str = 'jotty> ', multiline: bool = False, vi_mode: bool = False) -> None:
        """
        Initialize REPL engine.

        Args:
            command_registry: Command registry for completions
            history_file: Path to history file
            prompt_text: Prompt text to display
            multiline: Enable multiline input
            vi_mode: Use vi key bindings
        """
        self.command_registry = command_registry
        self.prompt_text = prompt_text
        self.multiline = multiline
        self.vi_mode = vi_mode

        # History
        self.history = HistoryManager(history_file)

        # Completer
        if PROMPT_TOOLKIT_AVAILABLE:
            self.completer = CommandCompleter(command_registry)
        else:
            self.completer = SimpleCompleter(command_registry)

        # Footer hints
        self._footer_hints = FooterHints()

        # Prompt session
        self._session: Optional["PromptSession"] = None
        self._running = False

    def _create_session(self) -> Optional["PromptSession"]:
        """Create prompt_toolkit session."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return None

        # Style - colors for different UI elements
        style = Style.from_dict({
            "": "ansiwhite",           # Default text
            "completion-menu": "bg:ansiblack ansiwhite",
            "completion-menu.completion": "bg:ansiblack ansiwhite",
            "completion-menu.completion.current": "bg:ansicyan ansiblack",
            "completion-menu.meta.completion": "bg:ansiblack ansigray",
            "completion-menu.meta.completion.current": "bg:ansicyan ansiblack",
        })

        # Bottom toolbar callback
        def _bottom_toolbar() -> Any:
            return HTML(self._footer_hints.get_toolbar_text())

        # Create session
        session = PromptSession(
            history=self.history.get_prompt_toolkit_history(),
            completer=self.completer,
            style=style,
            multiline=self.multiline,
            vi_mode=self.vi_mode,
            enable_history_search=True,
            complete_while_typing=True,
            bottom_toolbar=_bottom_toolbar,
        )

        return session

    async def run(self, handler: Callable[[str], Awaitable[bool]], welcome_callback: Optional[Callable[[], None]] = None) -> Any:
        """
        Run the REPL loop.

        Args:
            handler: Async function to handle input. Returns True to continue, False to exit.
            welcome_callback: Optional callback to show welcome message
        """
        self._running = True

        # Show welcome
        if welcome_callback:
            welcome_callback()

        # Create session
        self._session = self._create_session()

        while self._running:
            try:
                # Get input
                user_input = await self._get_input()

                if user_input is None:
                    # EOF
                    break

                # Skip empty input
                if not user_input.strip():
                    continue

                # Add to history
                self.history.add(user_input)

                # Handle input
                should_continue = await handler(user_input)

                if not should_continue:
                    break

            except KeyboardInterrupt:
                print()  # New line after ^C
                continue

            except EOFError:
                break

            except Exception as e:
                logger.error(f"REPL error: {e}")
                if logger.level <= logging.DEBUG:
                    import traceback
                    traceback.print_exc()

        self._running = False

    async def _get_input(self) -> Optional[str]:
        """
        Get user input.

        Returns:
            User input string or None on EOF
        """
        if self._session:
            # Use prompt_toolkit
            try:
                # prompt_toolkit's prompt_async
                if hasattr(self._session, "prompt_async"):
                    # Use plain prompt - styling done via session style
                    return await self._session.prompt_async(self.prompt_text)
                else:
                    # Fallback to sync prompt in thread
                    import asyncio
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self._session.prompt(self.prompt_text)
                    )
            except EOFError:
                return None
            except KeyboardInterrupt:
                raise

        else:
            # Fallback to basic input
            try:
                return input(self.prompt_text)
            except EOFError:
                return None

    def stop(self) -> Any:
        """Stop the REPL loop."""
        self._running = False

    def set_prompt(self, text: str) -> Any:
        """Set prompt text."""
        self.prompt_text = text

    def set_repl_state(self, state: REPLState) -> Any:
        """
        Set the REPL state to update footer hints.

        Args:
            state: New REPL state (INPUT, EXECUTING, REVIEWING, EXPORTING)
        """
        self._footer_hints.state = state

    def get_input_sync(self, prompt: str = None) -> str:
        """
        Get input synchronously (for simple prompts).

        Args:
            prompt: Optional custom prompt

        Returns:
            User input
        """
        prompt = prompt or self.prompt_text

        if self._session:
            return self._session.prompt(prompt)
        return input(prompt)


class SimpleREPL:
    """
    Simple REPL without prompt_toolkit.

    Fallback for environments without prompt_toolkit.
    """

    def __init__(self, command_registry: 'CommandRegistry', prompt_text: str = 'jotty> ') -> None:
        self.command_registry = command_registry
        self.prompt_text = prompt_text
        self._running = False

    async def run(self, handler: Callable[[str], Awaitable[bool]], welcome_callback: Optional[Callable[[], None]] = None) -> Any:
        """Run simple REPL loop."""
        self._running = True

        if welcome_callback:
            welcome_callback()

        while self._running:
            try:
                user_input = input(self.prompt_text)

                if not user_input.strip():
                    continue

                should_continue = await handler(user_input)

                if not should_continue:
                    break

            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

        self._running = False

    def stop(self) -> Any:
        """Stop REPL."""
        self._running = False
