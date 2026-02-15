"""
Unified Message Format
======================

JottyMessage provides a common message format across all interfaces:
- CLI (terminal)
- Telegram Bot
- Web UI

This enables cross-interface sync and consistent processing.
"""

from dataclasses import dataclass, field, asdict, fields as dataclass_fields
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
        """Convert to dictionary (excluding binary data). DRY: uses asdict."""
        return {k: v for k, v in asdict(self).items() if k != 'data'}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Create from dictionary. DRY: filters to valid fields only."""
        valid_fields = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_fields}
        # Set defaults for required fields if missing
        kwargs.setdefault("filename", "")
        kwargs.setdefault("content_type", "application/octet-stream")
        kwargs.setdefault("size", 0)
        return cls(**kwargs)


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
        Create from Telegram update. DRY: delegates to MessageAdapter.

        Args:
            update: Telegram Update object
            session_id: Optional session ID override

        Returns:
            JottyMessage instance
        """
        return MessageAdapter._from_telegram(update, session_id=session_id)

    @classmethod
    def from_web(
        cls,
        request_data: Dict[str, Any],
        user_id: str = "web_user",
        session_id: Optional[str] = None
    ) -> "JottyMessage":
        """
        Create from Web API request. DRY: delegates to MessageAdapter.

        Args:
            request_data: Request body dict with 'message', 'session_id', etc.
            user_id: User identifier (from auth or session)
            session_id: Session ID override

        Returns:
            JottyMessage instance
        """
        return MessageAdapter._from_web(request_data, user_id=user_id, session_id=session_id)

    @classmethod
    def from_cli(
        cls,
        text: str,
        session_id: str,
        user_id: str = "cli_user"
    ) -> "JottyMessage":
        """
        Create from CLI input. DRY: delegates to MessageAdapter.

        Args:
            text: User input text
            session_id: Current session ID
            user_id: User identifier

        Returns:
            JottyMessage instance
        """
        return MessageAdapter._from_cli(text, session_id=session_id, user_id=user_id)

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
        """Convert to dict, excluding None/empty values. DRY: uses asdict."""
        d = asdict(self)
        d['event_type'] = d['event_type'].value  # Enum to string

        # Core fields always included
        core_fields = {'event_type', 'event_id', 'source', 'timestamp'}

        # Filter out None and empty values, keep core fields
        result = {}
        for k, v in d.items():
            if k in core_fields:
                result[k] = v
            elif v not in (None, '', 0.0, {}):
                # Cap strings for serialization
                if k in ('output', 'error') and isinstance(v, str):
                    result[k] = v[:500]
                else:
                    result[k] = v

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InternalEvent":
        """Create from dict. DRY: uses dataclass fields for validation."""
        valid_fields = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_fields}

        # Handle enum conversion
        if 'event_type' in kwargs and isinstance(kwargs['event_type'], str):
            kwargs['event_type'] = EventType(kwargs['event_type'])

        # Set defaults for required fields
        kwargs.setdefault('event_type', EventType.AGENT_START)
        kwargs.setdefault('source', '')

        return cls(**kwargs)

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


# =============================================================================
# MESSAGE ADAPTER - DRY Message Conversion Pattern
# =============================================================================
# Unifies from_telegram, from_web, from_cli into a single strategy pattern


class MessageAdapter:
    """
    DRY adapter for converting external messages to JottyMessage.

    Eliminates duplication across from_telegram, from_web, from_cli methods.
    Uses strategy pattern for clean separation of concerns.

    Usage:
        # Single entry point
        msg = MessageAdapter.from_source(InterfaceType.TELEGRAM, telegram_update)

        # Or use existing methods (backwards compatible)
        msg = JottyMessage.from_telegram(telegram_update)
    """

    @staticmethod
    def from_source(
        source_type: InterfaceType,
        data: Any,
        **kwargs: Any
    ) -> JottyMessage:
        """
        Single entry point for message conversion.

        Args:
            source_type: InterfaceType enum
            data: Source-specific data (Update, dict, str)
            **kwargs: Additional arguments passed to converter

        Returns:
            JottyMessage instance
        """
        converters = {
            InterfaceType.TELEGRAM: MessageAdapter._from_telegram,
            InterfaceType.WEB: MessageAdapter._from_web,
            InterfaceType.CLI: MessageAdapter._from_cli,
        }

        converter = converters.get(source_type)
        if not converter:
            raise ValueError(f"Unsupported interface type: {source_type}")

        return converter(data, **kwargs)

    @staticmethod
    def _from_telegram(update: Any, session_id: Optional[str] = None) -> JottyMessage:
        """Internal: Convert Telegram update. Called by from_source."""
        message = update.message or update.edited_message
        if not message:
            raise ValueError("No message in update")

        chat_id = str(message.chat.id)
        user_id = str(message.from_user.id) if message.from_user else chat_id

        # Build attachments
        attachments = []
        if message.document:
            attachments.append(Attachment(
                filename=message.document.file_name or "document",
                content_type=message.document.mime_type or "application/octet-stream",
                size=message.document.file_size or 0,
                metadata={"file_id": message.document.file_id}
            ))
        if message.photo:
            photo = message.photo[-1]
            attachments.append(Attachment(
                filename=f"photo_{photo.file_unique_id}.jpg",
                content_type="image/jpeg",
                size=photo.file_size or 0,
                metadata={"file_id": photo.file_id}
            ))

        return JottyMessage(
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

    @staticmethod
    def _from_web(
        request_data: Dict[str, Any],
        user_id: str = "web_user",
        session_id: Optional[str] = None
    ) -> JottyMessage:
        """Internal: Convert web request. Called by from_source."""
        return JottyMessage(
            content=request_data.get("message", ""),
            interface=InterfaceType.WEB,
            user_id=user_id,
            session_id=session_id or request_data.get("session_id", str(uuid.uuid4())[:8]),
            metadata={
                "user_agent": request_data.get("user_agent"),
                "ip": request_data.get("ip"),
            },
        )

    @staticmethod
    def _from_cli(
        text: str,
        session_id: str,
        user_id: str = "cli_user"
    ) -> JottyMessage:
        """Internal: Convert CLI input. Called by from_source."""
        return JottyMessage(
            content=text,
            interface=InterfaceType.CLI,
            user_id=user_id,
            session_id=session_id,
        )
