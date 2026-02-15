"""
Abstract Interfaces
===================

Platform-agnostic interfaces that all renderers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, List, Optional

from .models import ChatSession, Error, Message, Status
from .state import ChatState, ChatStateMachine


class MessageRenderer(ABC):
    """
    Abstract message renderer.

    All platforms (Terminal, Telegram, Web, etc.) must implement this interface.
    """

    @abstractmethod
    def render_text(self, text: str) -> Any:
        """
        Render plain text.

        Args:
            text: Plain text to display

        Returns:
            Platform-specific rendered output
        """
        pass

    @abstractmethod
    def render_markdown(self, markdown: str) -> Any:
        """
        Render markdown content.

        Args:
            markdown: Markdown text

        Returns:
            Platform-specific rendered markdown
        """
        pass

    @abstractmethod
    def render_code(self, code: str, language: str = "python") -> Any:
        """
        Render code block with syntax highlighting.

        Args:
            code: Code to display
            language: Programming language for syntax highlighting

        Returns:
            Platform-specific rendered code
        """
        pass

    @abstractmethod
    def render_message(self, message: Message) -> Any:
        """
        Render complete message (with role, timestamp, etc.).

        Args:
            message: Message to render

        Returns:
            Platform-specific rendered message
        """
        pass

    @abstractmethod
    def render_message_list(self, messages: List[Message]) -> Any:
        """
        Render list of messages (chat history).

        Args:
            messages: List of messages

        Returns:
            Platform-specific rendered message list
        """
        pass

    @abstractmethod
    def update_streaming_message(self, message: Message, chunk: str) -> Any:
        """
        Update a streaming message with new chunk.

        Args:
            message: Original message being updated
            chunk: New text chunk to append

        Returns:
            Platform-specific update
        """
        pass

    @abstractmethod
    def clear_display(self) -> None:
        """Clear the display."""
        pass


class StatusRenderer(ABC):
    """Abstract status/progress renderer."""

    @abstractmethod
    def render_status(self, status: Status) -> Any:
        """
        Render status indicator.

        Args:
            status: Status to display

        Returns:
            Platform-specific rendered status
        """
        pass

    @abstractmethod
    def render_progress(self, progress: float, message: Optional[str] = None) -> Any:
        """
        Render progress bar/indicator.

        Args:
            progress: Progress (0.0-1.0)
            message: Optional progress message

        Returns:
            Platform-specific rendered progress
        """
        pass

    @abstractmethod
    def render_thinking(self, message: Optional[str] = None) -> Any:
        """
        Render thinking indicator (animated).

        Args:
            message: Optional thinking message

        Returns:
            Platform-specific thinking indicator
        """
        pass

    @abstractmethod
    def render_error(self, error: Error) -> Any:
        """
        Render error message.

        Args:
            error: Error to display

        Returns:
            Platform-specific rendered error
        """
        pass

    @abstractmethod
    def update_status(self, status: Status) -> None:
        """
        Update existing status display.

        Args:
            status: New status
        """
        pass

    @abstractmethod
    def clear_status(self) -> None:
        """Clear status display."""
        pass


class InputHandler(ABC):
    """Abstract input handler."""

    @abstractmethod
    async def get_input(self, prompt: str = "> ") -> Optional[str]:
        """
        Get user text input.

        Args:
            prompt: Prompt to display

        Returns:
            User input or None if cancelled
        """
        pass

    @abstractmethod
    async def get_multiline_input(self, prompt: str = "> ") -> Optional[str]:
        """
        Get multiline user input.

        Args:
            prompt: Prompt to display

        Returns:
            Multiline input or None if cancelled
        """
        pass

    @abstractmethod
    async def get_voice_input(self) -> Optional[bytes]:
        """
        Get voice input (audio recording).

        Returns:
            Audio data or None if cancelled
        """
        pass

    @abstractmethod
    async def confirm(self, message: str) -> bool:
        """
        Get yes/no confirmation.

        Args:
            message: Confirmation message

        Returns:
            True if confirmed, False otherwise
        """
        pass


class ChatInterface:
    """
    Unified chat interface using abstract renderers.

    This is the main class used by all platforms.
    Platforms provide their own MessageRenderer, StatusRenderer, InputHandler.
    """

    def __init__(
        self,
        message_renderer: MessageRenderer,
        status_renderer: StatusRenderer,
        input_handler: InputHandler,
        session: Optional[ChatSession] = None,
    ):
        """
        Initialize chat interface.

        Args:
            message_renderer: Platform-specific message renderer
            status_renderer: Platform-specific status renderer
            input_handler: Platform-specific input handler
            session: Optional existing session
        """
        self.message_renderer = message_renderer
        self.status_renderer = status_renderer
        self.input_handler = input_handler
        self.session = session or ChatSession(session_id="default")
        self.state_machine = ChatStateMachine()

        # Callbacks
        self._on_message_callback: Optional[Callable] = None
        self._on_state_change_callback: Optional[Callable] = None

    def add_message(self, message: Message) -> None:
        """
        Add message to session and render.

        Args:
            message: Message to add
        """
        self.session.add_message(message)
        self.message_renderer.render_message(message)

        if self._on_message_callback:
            self._on_message_callback(message)

    def update_streaming_message(self, message: Message, chunk: str) -> None:
        """
        Update streaming message with new chunk.

        Args:
            message: Message being updated
            chunk: New chunk to append
        """
        message.content += chunk
        self.message_renderer.update_streaming_message(message, chunk)

    def set_state(self, state: ChatState, **kwargs) -> None:
        """
        Transition to new state and update display.

        Args:
            state: New state
            **kwargs: State context data
        """
        if self.state_machine.transition(state, **kwargs):
            context = self.state_machine.get_context()
            status = Status(
                state=state.value,
                message=context.get_display_text(),
                progress=context.progress,
                details={
                    "skill_name": context.skill_name,
                    "agent_name": context.agent_name,
                    "swarm_id": context.swarm_id,
                },
            )
            self.status_renderer.update_status(status)

            if self._on_state_change_callback:
                self._on_state_change_callback(context)

    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update progress indicator.

        Args:
            progress: Progress (0.0-1.0)
            message: Optional progress message
        """
        self.state_machine.update_progress(progress)
        self.status_renderer.render_progress(progress, message)

    def show_error(self, error: Error) -> None:
        """
        Display error.

        Args:
            error: Error to display
        """
        self.state_machine.transition(ChatState.ERROR)
        self.status_renderer.render_error(error)

    def clear(self) -> None:
        """Clear display and reset state."""
        self.message_renderer.clear_display()
        self.status_renderer.clear_status()
        self.state_machine.reset()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get message history.

        Args:
            limit: Max number of messages

        Returns:
            List of messages
        """
        return self.session.get_messages(limit=limit)

    def on_message(self, callback: Callable[[Message], None]) -> None:
        """Register callback for new messages."""
        self._on_message_callback = callback

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes."""
        self._on_state_change_callback = callback
