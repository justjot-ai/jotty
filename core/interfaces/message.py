"""
Unified Message Format
======================

JottyMessage provides a common message format across all interfaces:
- CLI (terminal)
- Telegram Bot
- Web UI

This enables cross-interface sync and consistent processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import uuid


class InterfaceType(Enum):
    """Source interface for a message."""
    CLI = "cli"
    TELEGRAM = "telegram"
    WEB = "web"
    API = "api"  # Direct API calls


@dataclass
class Attachment:
    """File or media attachment."""
    filename: str
    content_type: str
    size: int
    data: Optional[bytes] = None
    url: Optional[str] = None  # Remote URL if not storing data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding binary data)."""
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "url": self.url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Create from dictionary."""
        return cls(
            filename=data.get("filename", ""),
            content_type=data.get("content_type", "application/octet-stream"),
            size=data.get("size", 0),
            url=data.get("url"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JottyMessage:
    """
    Unified message format for all Jotty interfaces.

    Provides a common structure for messages from:
    - CLI REPL
    - Telegram Bot
    - Web UI

    All messages are normalized to this format before processing
    and can be stored/synced across interfaces.
    """

    content: str
    interface: InterfaceType
    user_id: str
    session_id: str
    role: str = "user"  # user, assistant, system
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)

    # Optional reply context
    reply_to: Optional[str] = None  # message_id being replied to

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "interface": self.interface.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "attachments": [a.to_dict() for a in self.attachments],
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JottyMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())[:12]),
            content=data.get("content", ""),
            interface=InterfaceType(data.get("interface", "cli")),
            user_id=data.get("user_id", "unknown"),
            session_id=data.get("session_id", ""),
            role=data.get("role", "user"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metadata=data.get("metadata", {}),
            attachments=[Attachment.from_dict(a) for a in data.get("attachments", [])],
            reply_to=data.get("reply_to"),
        )

    @classmethod
    def from_telegram(
        cls,
        update: Any,  # telegram.Update object
        session_id: Optional[str] = None
    ) -> "JottyMessage":
        """
        Create from Telegram update.

        Args:
            update: Telegram Update object
            session_id: Optional session ID override

        Returns:
            JottyMessage instance
        """
        message = update.message or update.edited_message

        if not message:
            raise ValueError("No message in update")

        chat_id = str(message.chat.id)
        user_id = str(message.from_user.id) if message.from_user else chat_id

        # Build attachments from documents/photos
        attachments = []
        if message.document:
            attachments.append(Attachment(
                filename=message.document.file_name or "document",
                content_type=message.document.mime_type or "application/octet-stream",
                size=message.document.file_size or 0,
                metadata={"file_id": message.document.file_id}
            ))
        if message.photo:
            # Get largest photo
            photo = message.photo[-1]
            attachments.append(Attachment(
                filename=f"photo_{photo.file_unique_id}.jpg",
                content_type="image/jpeg",
                size=photo.file_size or 0,
                metadata={"file_id": photo.file_id}
            ))

        return cls(
            content=message.text or message.caption or "",
            interface=InterfaceType.TELEGRAM,
            user_id=user_id,
            session_id=session_id or f"tg_{chat_id}",
            metadata={
                "chat_id": chat_id,
                "message_id": message.message_id,
                "chat_type": message.chat.type,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
            },
            attachments=attachments,
        )

    @classmethod
    def from_web(
        cls,
        request_data: Dict[str, Any],
        user_id: str = "web_user",
        session_id: Optional[str] = None
    ) -> "JottyMessage":
        """
        Create from Web API request.

        Args:
            request_data: Request body dict with 'message', 'session_id', etc.
            user_id: User identifier (from auth or session)
            session_id: Session ID override

        Returns:
            JottyMessage instance
        """
        return cls(
            content=request_data.get("message", ""),
            interface=InterfaceType.WEB,
            user_id=user_id,
            session_id=session_id or request_data.get("session_id", str(uuid.uuid4())[:8]),
            metadata={
                "user_agent": request_data.get("user_agent"),
                "ip": request_data.get("ip"),
            },
        )

    @classmethod
    def from_cli(
        cls,
        text: str,
        session_id: str,
        user_id: str = "cli_user"
    ) -> "JottyMessage":
        """
        Create from CLI input.

        Args:
            text: User input text
            session_id: Current session ID
            user_id: User identifier

        Returns:
            JottyMessage instance
        """
        return cls(
            content=text,
            interface=InterfaceType.CLI,
            user_id=user_id,
            session_id=session_id,
        )

    @classmethod
    def assistant_response(
        cls,
        content: str,
        original_message: "JottyMessage",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "JottyMessage":
        """
        Create assistant response to a user message.

        Args:
            content: Response content
            original_message: The message being responded to
            metadata: Optional additional metadata

        Returns:
            JottyMessage instance with role='assistant'
        """
        return cls(
            content=content,
            interface=original_message.interface,
            user_id="assistant",
            session_id=original_message.session_id,
            role="assistant",
            reply_to=original_message.message_id,
            metadata=metadata or {},
        )


# =============================================================================
# INTERNAL EVENT PROTOCOL (Cline protobuf pattern, KISS version)
# =============================================================================
# Typed dataclasses for inter-component communication.
# Replaces raw dicts in kwargs paths. Each event is self-documenting.
# No protobuf — just typed Python dataclasses with to_dict/from_dict.


class EventType(Enum):
    """Internal event types for component communication."""
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    LEARNING_UPDATE = "learning_update"
    PROGRESS_UPDATE = "progress_update"
    GUARD_DECISION = "guard_decision"


@dataclass
class InternalEvent:
    """
    Typed internal event for component-to-component communication.

    Replaces raw dicts (kwargs) between agent_runner, swarm_manager,
    learning_pipeline, etc. Each field is typed and documented.
    """
    event_type: EventType
    source: str              # Component name (e.g., "agent_runner", "swarm_manager")
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Payload — type depends on event_type
    agent_name: str = ""
    goal: str = ""
    success: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None
    tool_name: str = ""
    trust_level: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'source': self.source,
            'timestamp': self.timestamp,
            'agent_name': self.agent_name,
            'goal': self.goal,
        }
        # Only include non-default fields to keep messages lean
        if self.success is not None:
            d['success'] = self.success
        if self.output:
            d['output'] = self.output[:500]  # Cap for serialization
        if self.error:
            d['error'] = self.error[:500]
        if self.tool_name:
            d['tool_name'] = self.tool_name
        if self.trust_level:
            d['trust_level'] = self.trust_level
        if self.execution_time:
            d['execution_time'] = self.execution_time
        if self.metadata:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InternalEvent":
        return cls(
            event_type=EventType(data.get('event_type', 'agent_start')),
            event_id=data.get('event_id', ''),
            source=data.get('source', ''),
            timestamp=data.get('timestamp', 0),
            agent_name=data.get('agent_name', ''),
            goal=data.get('goal', ''),
            success=data.get('success'),
            output=data.get('output'),
            error=data.get('error'),
            tool_name=data.get('tool_name', ''),
            trust_level=data.get('trust_level', ''),
            execution_time=data.get('execution_time', 0),
            metadata=data.get('metadata', {}),
        )

    # Convenience constructors
    @classmethod
    def tool_call(cls, tool_name: str, trust_level: str, agent: str = "") -> "InternalEvent":
        return cls(event_type=EventType.TOOL_CALL, source="agent_runner",
                   tool_name=tool_name, trust_level=trust_level, agent_name=agent)

    @classmethod
    def agent_complete(cls, agent: str, goal: str, success: bool,
                       output: str = "", time: float = 0) -> "InternalEvent":
        return cls(event_type=EventType.AGENT_COMPLETE, source="agent_runner",
                   agent_name=agent, goal=goal, success=success,
                   output=output, execution_time=time)

    @classmethod
    def progress_update(cls, agent: str, progress_data: Dict) -> "InternalEvent":
        return cls(event_type=EventType.PROGRESS_UPDATE, source="agent_runner",
                   agent_name=agent, metadata=progress_data)
