"""
Terminal Renderer
=================

Rich-based renderer for terminal/TUI.
"""

import asyncio
from typing import Any, List, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from rich import box
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.text import Text

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    Console = None

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ..interface import InputHandler, MessageRenderer, StatusRenderer
from ..models import Error, Message, Status


class TerminalMessageRenderer(MessageRenderer):
    """
    Terminal message renderer using Rich.

    Renders messages with:
    - Markdown formatting
    - Syntax-highlighted code blocks
    - Role-based colors (user=blue, assistant=green)
    - Timestamps
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize terminal renderer.

        Args:
            console: Rich console (creates one if not provided)
        """
        if not DEPS_AVAILABLE:
            raise ImportError("rich library required for TerminalMessageRenderer")

        self.console = console or Console()
        self._last_message_id: Optional[str] = None

    def render_text(self, text: str) -> Any:
        """Render plain text."""
        self.console.print(text)

    def render_markdown(self, markdown: str) -> Any:
        """Render markdown content."""
        md = Markdown(markdown)
        self.console.print(md)

    def render_code(self, code: str, language: str = "python") -> Any:
        """Render code block with syntax highlighting."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def render_message(self, message: Message) -> Any:
        """Render complete message."""
        # Get role color
        role_colors = {
            "user": "blue",
            "assistant": "green",
            "system": "yellow",
        }
        color = role_colors.get(message.role, "white")

        # Get status icon
        icon = message.get_status_icon()
        prefix = f"{icon} " if icon else ""

        # Format role and timestamp
        timestamp = message.timestamp.strftime("%H:%M:%S")
        header = f"[{color}]{prefix}{message.role.capitalize()}[/{color}] [{timestamp}]"

        # Render based on format
        if message.format.value == "markdown":
            md = Markdown(message.content)
            panel = Panel(
                md,
                title=header,
                border_style=color,
                box=box.ROUNDED,
            )
            self.console.print(panel)
        else:
            # Plain text
            self.console.print(f"{header}\n{message.content}")

        self._last_message_id = message.id

    def render_message_list(self, messages: List[Message]) -> Any:
        """Render list of messages."""
        for message in messages:
            if not message.hidden:
                self.render_message(message)

    def update_streaming_message(self, message: Message, chunk: str) -> Any:
        """Update a streaming message with new chunk."""
        # For streaming, just print the chunk without newline
        self.console.print(chunk, end="")

    def clear_display(self) -> None:
        """Clear the terminal display."""
        self.console.clear()


class TerminalStatusRenderer(StatusRenderer):
    """
    Terminal status renderer using Rich progress bars.

    Displays:
    - Thinking indicators (spinner)
    - Progress bars
    - Status messages
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize status renderer.

        Args:
            console: Rich console
        """
        if not DEPS_AVAILABLE:
            raise ImportError("rich library required for TerminalStatusRenderer")

        self.console = console or Console()
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None

    def render_status(self, status: Status) -> Any:
        """Render status indicator."""
        icon = status.icon or "⚙️"
        message = status.message or status.state
        self.console.print(f"{icon} {message}", style="italic")

    def render_progress(self, progress: float, message: Optional[str] = None) -> Any:
        """Render progress bar."""
        if not self._progress:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                message or "Processing...",
                total=100,
            )

        if self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=int(progress * 100),
                description=message or "Processing...",
            )

    def render_thinking(self, message: Optional[str] = None) -> Any:
        """Render thinking indicator."""
        text = Text(f"🤔 {message or 'Thinking...'}", style="italic cyan")
        self.console.print(text)

    def render_error(self, error: Error) -> Any:
        """Render error message."""
        panel = Panel(
            f"[red]{error.message}[/red]",
            title="❌ Error",
            border_style="red",
            box=box.HEAVY,
        )
        self.console.print(panel)

        if error.traceback:
            self.console.print(f"\n[dim]{error.traceback}[/dim]")

    def update_status(self, status: Status) -> None:
        """Update existing status display."""
        self.render_status(status)

    def clear_status(self) -> None:
        """Clear status display."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None


class TerminalInputHandler(InputHandler):
    """
    Terminal input handler using prompt_toolkit.

    Provides:
    - Text input with autocomplete
    - Multiline input
    - Confirmation dialogs
    """

    def __init__(self):
        """Initialize input handler."""
        if not DEPS_AVAILABLE:
            raise ImportError("prompt_toolkit required for TerminalInputHandler")

        self._session: Optional[PromptSession] = None

    def _get_session(self) -> PromptSession:
        """Get or create prompt session."""
        if not self._session:
            self._session = PromptSession()
        return self._session

    async def get_input(self, prompt: str = "> ") -> Optional[str]:
        """Get user text input."""
        try:
            session = self._get_session()
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: session.prompt(HTML(f"<green>{prompt}</green>")),
            )
            return result
        except (KeyboardInterrupt, EOFError):
            return None

    async def get_multiline_input(self, prompt: str = "> ") -> Optional[str]:
        """Get multiline user input."""
        try:
            session = self._get_session()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: session.prompt(
                    HTML(f"<green>{prompt}</green>"),
                    multiline=True,
                ),
            )
            return result
        except (KeyboardInterrupt, EOFError):
            return None

    async def get_voice_input(self) -> Optional[bytes]:
        """Get voice input (not supported in terminal)."""
        print("Voice input not supported in terminal")
        return None

    async def confirm(self, message: str) -> bool:
        """Get yes/no confirmation."""
        try:
            session = self._get_session()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: session.prompt(HTML(f"<yellow>{message} (y/n): </yellow>")),
            )
            return result.lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False
