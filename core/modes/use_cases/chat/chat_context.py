"""
Chat Context Management

Handles conversation history and context for chat use cases.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Structured chat message."""

    role: str  # user, assistant, system, tool
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ChatContext:
    """
    Manages conversation context for chat interactions.
    """

    def __init__(self, max_history: int = 50, system_prompt: Optional[str] = None) -> None:
        """
        Initialize chat context.

        Args:
            max_history: Maximum number of messages to keep in history
            system_prompt: Optional system prompt
        """
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.messages: List[ChatMessage] = []

        # Add system message if provided
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the conversation."""
        message = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)

        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message if present
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]

            # Keep most recent messages
            keep_count = self.max_history - len(system_msgs)
            self.messages = system_msgs + other_msgs[-keep_count:]

    def get_history(self) -> List[ChatMessage]:
        """Get conversation history."""
        return self.messages.copy()

    def get_formatted_history(self) -> str:
        """Get formatted conversation history as string."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get recent messages (excluding system messages)."""
        non_system = [m for m in self.messages if m.role != "system"]
        return non_system[-count:]

    def clear(self) -> None:
        """Clear conversation history (keeps system message)."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        self.messages = system_msgs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_history": self.max_history,
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatContext":
        """Create from dictionary."""
        context = cls(
            max_history=data.get("max_history", 50), system_prompt=data.get("system_prompt")
        )
        context.messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        return context
