"""
Shared Data Models
==================

Unified message and content models used across all platforms.
"""

import os

# Import SDK types
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from Jotty.sdk import ResponseFormat, SDKEventType


class AttachmentType(Enum):
    """Type of attachment."""

    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class Attachment:
    """Message attachment (file, image, audio, etc.)."""

    type: AttachmentType
    url: Optional[str] = None
    path: Optional[Path] = None
    name: Optional[str] = None
    size: Optional[int] = None  # bytes
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Get display name for attachment."""
        if self.name:
            return self.name
        if self.path:
            return self.path.name
        if self.url:
            return self.url.split("/")[-1]
        return f"{self.type.value.capitalize()} attachment"


@dataclass
class Message:
    """
    Unified message model for all platforms.

    This model captures ALL information needed to display
    a message consistently across Terminal, Telegram, Web, etc.
    """

    # Core fields
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    format: ResponseFormat = ResponseFormat.MARKDOWN

    # Event metadata (for status/progress messages)
    event_type: Optional[SDKEventType] = None
    skill_name: Optional[str] = None
    agent_name: Optional[str] = None
    swarm_id: Optional[str] = None
    progress: Optional[float] = None  # 0.0-1.0
    progress_total: Optional[int] = None  # e.g., "Step 3/10"
    progress_current: Optional[int] = None

    # Attachments
    attachments: List[Attachment] = field(default_factory=list)

    # Rendering hints
    ephemeral: bool = False  # Delete after display (e.g., "typing..." indicator)
    priority: int = 0  # Higher priority = display first
    collapsible: bool = False  # Can be collapsed in UI
    hidden: bool = False  # Hidden by default (show on expand)

    # Metadata
    id: Optional[str] = None
    parent_id: Optional[str] = None  # For threading
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_status(self) -> bool:
        """Check if this is a status message (thinking, planning, etc.)."""
        return self.event_type in (
            SDKEventType.THINKING,
            SDKEventType.PLANNING,
            SDKEventType.SKILL_START,
            SDKEventType.SKILL_PROGRESS,
            SDKEventType.AGENT_START,
            SDKEventType.SWARM_AGENT_START,
        )

    def is_streaming(self) -> bool:
        """Check if this is a streaming message."""
        return self.event_type in (SDKEventType.STREAM, SDKEventType.DELTA)

    def is_error(self) -> bool:
        """Check if this is an error message."""
        return self.event_type == SDKEventType.ERROR

    def is_complete(self) -> bool:
        """Check if this marks completion."""
        return self.event_type in (
            SDKEventType.COMPLETE,
            SDKEventType.SKILL_COMPLETE,
            SDKEventType.AGENT_COMPLETE,
            SDKEventType.SWARM_AGENT_COMPLETE,
        )

    def get_status_icon(self) -> str:
        """Get emoji/icon for status messages."""
        if not self.event_type:
            return ""

        icon_map = {
            SDKEventType.START: "▶️",
            SDKEventType.THINKING: "🤔",
            SDKEventType.PLANNING: "📋",
            SDKEventType.SKILL_START: "🔧",
            SDKEventType.SKILL_PROGRESS: "⏳",
            SDKEventType.SKILL_COMPLETE: "✅",
            SDKEventType.AGENT_START: "🤖",
            SDKEventType.AGENT_COMPLETE: "✓",
            SDKEventType.MEMORY_RECALL: "🧠",
            SDKEventType.MEMORY_STORE: "💾",
            SDKEventType.VALIDATION_START: "🔍",
            SDKEventType.VALIDATION_COMPLETE: "✓",
            SDKEventType.LEARNING_UPDATE: "📚",
            SDKEventType.VOICE_STT_START: "🎤",
            SDKEventType.VOICE_TTS_START: "🔊",
            SDKEventType.SWARM_AGENT_START: "🐝",
            SDKEventType.SWARM_COORDINATION: "🔀",
            SDKEventType.ERROR: "❌",
            SDKEventType.COMPLETE: "✅",
        }

        return icon_map.get(self.event_type, "•")

    def get_progress_text(self) -> Optional[str]:
        """Get progress text (e.g., '45%' or 'Step 3/10')."""
        if self.progress is not None:
            return f"{int(self.progress * 100)}%"
        if self.progress_current and self.progress_total:
            return f"Step {self.progress_current}/{self.progress_total}"
        return None


@dataclass
class Status:
    """Status information (thinking, planning, executing, etc.)."""

    state: str  # ChatState value
    message: Optional[str] = None
    icon: Optional[str] = None
    progress: Optional[float] = None  # 0.0-1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Error:
    """Error information."""

    message: str
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """Chat session data."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_messages(
        self, role: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Message]:
        """Get messages, optionally filtered by role and limited."""
        filtered = self.messages
        if role:
            filtered = [m for m in filtered if m.role == role]
        if limit:
            filtered = filtered[-limit:]
        return filtered

    def clear_ephemeral(self) -> None:
        """Remove ephemeral messages (typing indicators, etc.)."""
        self.messages = [m for m in self.messages if not m.ephemeral]
        self.updated_at = datetime.now()
