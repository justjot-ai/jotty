"""
Unified Message Format
======================

JottyMessage provides a common message format across all interfaces:
- CLI (terminal)
- Telegram Bot
- Web UI

This enables cross-interface sync and consistent processing.
"""

import uuid
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# =============================================================================
# NO_REPLY SENTINEL — signals "process but don't send response to user"
# =============================================================================
NO_REPLY = object()


def is_no_reply(result: Any) -> bool:
    """Check if a result is the NO_REPLY sentinel (identity check)."""
    return result is NO_REPLY


# =============================================================================
# STRUCTURED SESSION KEY — {channel}:{user}:{thread}
# =============================================================================


@dataclass(frozen=True)
class SessionKey:
    """
    Structured session identifier: {channel}:{user_id}:{thread_id}.

    Replaces opaque session_id strings with structured keys for better
    routing, debugging, and per-session operations (like lane queuing).

    Backward compatible: plain strings parse as unknown:{raw}:default.
    """

    channel: str
    user_id: str
    thread_id: str = "default"

    @property
    def raw(self) -> str:
        """Canonical string form: channel:user_id:thread_id."""
        return f"{self.channel}:{self.user_id}:{self.thread_id}"

    @classmethod
    def build(cls, channel: str, user_id: str, thread_id: str = "default") -> "SessionKey":
        """Build a SessionKey from components."""
        return cls(channel=channel, user_id=user_id, thread_id=thread_id)

    @classmethod
    def parse(cls, raw_string: str) -> "SessionKey":
        """
        Parse a raw session_id string into a SessionKey.

        Handles structured format (channel:user:thread) and legacy plain strings.
        """
        if not raw_string:
            return cls(channel="unknown", user_id="unknown", thread_id="default")

        parts = raw_string.split(":", 2)
        if len(parts) == 3:
            return cls(channel=parts[0], user_id=parts[1], thread_id=parts[2])
        elif len(parts) == 2:
            return cls(channel=parts[0], user_id=parts[1], thread_id="default")
        else:
            # Legacy: plain string → unknown:{raw}:default
            return cls(channel="unknown", user_id=raw_string, thread_id="default")

    def __str__(self) -> str:
        return self.raw

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SessionKey):
            return self.raw == other.raw
        if isinstance(other, str):
            return self.raw == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.raw)


def _make_interface_type() -> type:
    """
    InterfaceType is an alias for ChannelType (single source of truth).

    Backward compatible: InterfaceType.CLI, .TELEGRAM, .WEB all work.
    New code should use ChannelType directly.
    """
    try:
        from Jotty.core.infrastructure.foundation.types.sdk_types import ChannelType
        return ChannelType
    except ImportError:
        from enum import Enum as _Enum

        class _Fallback(_Enum):
            CLI = "cli"
            TELEGRAM = "telegram"
            WEB = "web"
            API = "api"

        return _Fallback


InterfaceType = _make_interface_type()


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
        return {k: v for k, v in asdict(self).items() if k != "data"}

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

    # NO_REPLY: if True, process but suppress output to user
    suppress_output: bool = False

    @property
    def session_key(self) -> SessionKey:
        """Structured session key parsed from session_id."""
        return SessionKey.parse(self.session_id)

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
            "suppress_output": self.suppress_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JottyMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())[:12]),
            content=data.get("content", ""),
            interface=InterfaceType.from_string(data.get("interface", "cli")),
            user_id=data.get("user_id", "unknown"),
            session_id=data.get("session_id", ""),
            role=data.get("role", "user"),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
            attachments=[Attachment.from_dict(a) for a in data.get("attachments", [])],
            reply_to=data.get("reply_to"),
            suppress_output=data.get("suppress_output", False),
        )

    @classmethod
    def from_telegram(
        cls, update: Any, session_id: Optional[str] = None  # telegram.Update object
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
        session_id: Optional[str] = None,
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
    def from_cli(cls, text: str, session_id: str, user_id: str = "cli_user") -> "JottyMessage":
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
        metadata: Optional[Dict[str, Any]] = None,
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
    source: str  # Component name (e.g., "agent_runner", "swarm_manager")
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
        d["event_type"] = d["event_type"].value  # Enum to string

        # Core fields always included
        core_fields = {"event_type", "event_id", "source", "timestamp"}

        # Filter out None and empty values, keep core fields
        result = {}
        for k, v in d.items():
            if k in core_fields:
                result[k] = v
            elif v not in (None, "", 0.0, {}):
                # Cap strings for serialization
                if k in ("output", "error") and isinstance(v, str):
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
        if "event_type" in kwargs and isinstance(kwargs["event_type"], str):
            kwargs["event_type"] = EventType(kwargs["event_type"])

        # Set defaults for required fields
        kwargs.setdefault("event_type", EventType.AGENT_START)
        kwargs.setdefault("source", "")

        return cls(**kwargs)

    # Convenience constructors
    @classmethod
    def tool_call(cls, tool_name: str, trust_level: str, agent: str = "") -> "InternalEvent":
        return cls(
            event_type=EventType.TOOL_CALL,
            source="agent_runner",
            tool_name=tool_name,
            trust_level=trust_level,
            agent_name=agent,
        )

    @classmethod
    def agent_complete(
        cls, agent: str, goal: str, success: bool, output: str = "", time: float = 0
    ) -> "InternalEvent":
        return cls(
            event_type=EventType.AGENT_COMPLETE,
            source="agent_runner",
            agent_name=agent,
            goal=goal,
            success=success,
            output=output,
            execution_time=time,
        )

    @classmethod
    def progress_update(cls, agent: str, progress_data: Dict) -> "InternalEvent":
        return cls(
            event_type=EventType.PROGRESS_UPDATE,
            source="agent_runner",
            agent_name=agent,
            metadata=progress_data,
        )


# =============================================================================
# MESSAGE ADAPTER - DRY Message Conversion Pattern
# =============================================================================
# Unifies from_telegram, from_web, from_cli into a single strategy pattern


class MessageAdapter:
    """
    DRY adapter for converting external messages to JottyMessage.

    All channels funnel through from_channel() which builds a JottyMessage
    from a simple dict. Only Telegram has special handling (SDK-specific
    Update objects). Every other channel (Slack, Discord, WhatsApp, Signal,
    X, LinkedIn, Mastodon, Bluesky, Matrix, Teams, etc.) uses the same
    generic dict->JottyMessage path.

    Usage:
        # Generic (works for ALL channels):
        msg = MessageAdapter.from_channel("telegram", {
            "content": "hello", "user_id": "123", "channel_id": "456"
        })

        # Backward compatible:
        msg = JottyMessage.from_telegram(telegram_update)
        msg = MessageAdapter.from_source(InterfaceType.TELEGRAM, update)
    """

    # Channel ID prefixes for session_id generation
    _CHANNEL_PREFIX = {
        "telegram": "tg", "whatsapp": "wa", "slack": "sl", "discord": "dc",
        "signal": "sg", "imessage": "im", "teams": "ms", "google_chat": "gc",
        "matrix": "mx", "x": "x", "linkedin": "li", "mastodon": "md",
        "bluesky": "bs", "reddit": "rd", "web": "web", "cli": "cli",
    }

    @staticmethod
    def from_channel(
        channel: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> JottyMessage:
        """
        Universal entry point: convert any channel's message dict to JottyMessage.

        This is the ONLY conversion path new channels need. The dict must contain:
          - content (str): message text
          - user_id (str): sender identifier
          - channel_id (str): chat/room/thread identifier

        Optional fields: user_name, message_id, attachments, metadata, reply_to.
        """
        channel_type = InterfaceType.from_string(channel)
        prefix = MessageAdapter._CHANNEL_PREFIX.get(channel, channel[:3])
        cid = data.get("channel_id", "default")

        return JottyMessage(
            content=data.get("content", ""),
            interface=channel_type,
            user_id=data.get("user_id", "unknown"),
            session_id=session_id or f"{prefix}_{cid}",
            metadata={
                "channel_id": cid,
                "message_id": data.get("message_id"),
                "user_name": data.get("user_name"),
                **data.get("metadata", {}),
            },
            attachments=[
                Attachment.from_dict(a) for a in data.get("attachments", [])
            ],
            reply_to=data.get("reply_to"),
        )

    @staticmethod
    def from_source(source_type: "InterfaceType", data: Any, **kwargs: Any) -> JottyMessage:
        """
        Backward-compatible entry point.

        Routes Telegram updates to the SDK-specific converter;
        everything else goes through from_channel().
        """
        if hasattr(source_type, "value") and source_type.value == "telegram" and not isinstance(data, dict):
            return MessageAdapter._from_telegram(data, **kwargs)
        if isinstance(data, str):
            return MessageAdapter._from_cli(data, **kwargs)
        if isinstance(data, dict):
            return MessageAdapter.from_channel(
                source_type.value if hasattr(source_type, "value") else str(source_type),
                data,
                session_id=kwargs.get("session_id"),
            )
        raise ValueError(f"Unsupported data type for {source_type}: {type(data)}")

    @staticmethod
    def _from_telegram(update: Any, session_id: Optional[str] = None) -> JottyMessage:
        """Convert Telegram SDK Update object (python-telegram-bot specific)."""
        message = update.message or update.edited_message
        if not message:
            raise ValueError("No message in update")

        chat_id = str(message.chat.id)
        user_id = str(message.from_user.id) if message.from_user else chat_id

        attachments = []
        if message.document:
            attachments.append(
                Attachment(
                    filename=message.document.file_name or "document",
                    content_type=message.document.mime_type or "application/octet-stream",
                    size=message.document.file_size or 0,
                    metadata={"file_id": message.document.file_id},
                )
            )
        if message.photo:
            photo = message.photo[-1]
            attachments.append(
                Attachment(
                    filename=f"photo_{photo.file_unique_id}.jpg",
                    content_type="image/jpeg",
                    size=photo.file_size or 0,
                    metadata={"file_id": photo.file_id},
                )
            )

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
        request_data: Dict[str, Any], user_id: str = "web_user", session_id: Optional[str] = None
    ) -> JottyMessage:
        """Convert web request dict."""
        return MessageAdapter.from_channel("web", {
            "content": request_data.get("message", ""),
            "user_id": user_id,
            "channel_id": request_data.get("session_id", str(uuid.uuid4())[:8]),
            "metadata": {
                "user_agent": request_data.get("user_agent"),
                "ip": request_data.get("ip"),
            },
        }, session_id=session_id)

    @staticmethod
    def _from_cli(text: str, session_id: str = "", user_id: str = "cli_user") -> JottyMessage:
        """Convert CLI text input."""
        return JottyMessage(
            content=text,
            interface=InterfaceType.CLI,
            user_id=user_id,
            session_id=session_id,
        )
