"""
SDK Types - Core data structures for world-class SDK
=====================================================

Provides unified types for:
- ExecutionContext: Unified context flowing through all layers
- SDKEvent: Event system for streaming and callbacks
- SDKSession: Persistent session management
- SDKResponse: Standardized response format
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

# =============================================================================
# ENUMS
# =============================================================================


class ExecutionMode(Enum):
    """Execution mode for requests."""

    CHAT = "chat"  # Conversational, single-turn
    WORKFLOW = "workflow"  # Multi-step autonomous execution
    AGENT = "agent"  # Direct agent invocation
    SKILL = "skill"  # Direct skill execution
    STREAM = "stream"  # Streaming response
    VOICE = "voice"  # Voice interaction (STT/TTS)
    SWARM = "swarm"  # Multi-agent swarm coordination


class ChannelType(Enum):
    """Source channel for requests."""

    CLI = "cli"  # Command-line interface
    WEB = "web"  # Web UI
    SDK = "sdk"  # SDK client (programmatic)
    TELEGRAM = "telegram"  # Telegram bot
    SLACK = "slack"  # Slack integration
    DISCORD = "discord"  # Discord bot
    WHATSAPP = "whatsapp"  # WhatsApp Business
    WEBSOCKET = "websocket"  # WebSocket connection
    HTTP = "http"  # Generic HTTP
    CUSTOM = "custom"  # Custom channel


class SDKEventType(Enum):
    """Types of SDK events for streaming/callbacks."""

    # Lifecycle events
    START = "start"  # Execution started
    COMPLETE = "complete"  # Execution completed
    ERROR = "error"  # Error occurred

    # Processing events
    THINKING = "thinking"  # Agent is reasoning
    PLANNING = "planning"  # Creating execution plan

    # Skill events
    SKILL_START = "skill_start"  # Skill execution starting
    SKILL_PROGRESS = "skill_progress"  # Skill progress update
    SKILL_COMPLETE = "skill_complete"  # Skill execution completed

    # Output events
    STREAM = "stream"  # Streaming text chunk
    DELTA = "delta"  # Incremental update

    # Agent events
    AGENT_START = "agent_start"  # Agent started
    AGENT_COMPLETE = "agent_complete"  # Agent completed

    # Memory events
    MEMORY_RECALL = "memory_recall"  # Retrieved from memory
    MEMORY_STORE = "memory_store"  # Stored to memory

    # Validation events
    VALIDATION_START = "validation_start"  # Validation started
    VALIDATION_COMPLETE = "validation_complete"  # Validation completed

    # Learning events
    LEARNING_UPDATE = "learning_update"  # Learning state updated

    # Voice events (modalities)
    VOICE_STT_START = "voice_stt_start"  # STT transcription started
    VOICE_STT_COMPLETE = "voice_stt_complete"  # STT transcription completed
    VOICE_TTS_START = "voice_tts_start"  # TTS synthesis started
    VOICE_TTS_CHUNK = "voice_tts_chunk"  # TTS audio chunk available
    VOICE_TTS_COMPLETE = "voice_tts_complete"  # TTS synthesis completed

    # Swarm events (multi-agent coordination)
    SWARM_AGENT_START = "swarm_agent_start"  # Swarm agent started
    SWARM_AGENT_COMPLETE = "swarm_agent_complete"  # Swarm agent completed
    SWARM_COORDINATION = "swarm_coordination"  # Swarm coordination event


class ResponseFormat(Enum):
    """Output format for responses."""

    TEXT = "text"  # Plain text
    MARKDOWN = "markdown"  # Markdown formatted
    JSON = "json"  # JSON data
    HTML = "html"  # HTML content
    A2UI = "a2ui"  # Agent-to-UI widgets


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass
class ExecutionContext:
    """
    Unified context that flows through all execution layers.

    This is the core data structure that enables:
    - Channel-aware execution (customize behavior per channel)
    - Session persistence across channels
    - User tracking and preferences
    - Memory integration
    - Mode routing

    Usage:
        context = ExecutionContext(
            mode=ExecutionMode.CHAT,
            channel=ChannelType.TELEGRAM,
            session_id="user-123-telegram",
            user_id="user-123"
        )
        result = await mode_router.execute(request, context)
    """

    # Required fields
    mode: ExecutionMode
    channel: ChannelType

    # Session and user
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    # Request tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Channel-specific data
    channel_id: Optional[str] = None  # Chat/channel ID
    message_id: Optional[str] = None  # Original message ID
    reply_to: Optional[str] = None  # Thread/reply context

    # Execution configuration
    timeout: float = 300.0  # Execution timeout in seconds
    max_steps: int = 10  # Max workflow steps
    streaming: bool = False  # Enable streaming
    response_format: ResponseFormat = ResponseFormat.MARKDOWN

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    # Callbacks (set at runtime, not serialized)
    event_callback: Optional[Callable] = field(default=None, repr=False)
    status_callback: Optional[Callable] = field(default=None, repr=False)
    stream_callback: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "channel": self.channel.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "channel_id": self.channel_id,
            "message_id": self.message_id,
            "reply_to": self.reply_to,
            "timeout": self.timeout,
            "max_steps": self.max_steps,
            "streaming": self.streaming,
            "response_format": self.response_format.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """Create from dictionary."""
        return cls(
            mode=ExecutionMode(data.get("mode", "chat")),
            channel=ChannelType(data.get("channel", "sdk")),
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            request_id=data.get("request_id", str(uuid.uuid4())),
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
            ),
            channel_id=data.get("channel_id"),
            message_id=data.get("message_id"),
            reply_to=data.get("reply_to"),
            timeout=data.get("timeout", 300.0),
            max_steps=data.get("max_steps", 10),
            streaming=data.get("streaming", False),
            response_format=ResponseFormat(data.get("response_format", "markdown")),
            metadata=data.get("metadata", {}),
        )

    def with_mode(self, mode: ExecutionMode) -> "ExecutionContext":
        """Return a copy with different mode."""
        return ExecutionContext(
            mode=mode,
            channel=self.channel,
            session_id=self.session_id,
            user_id=self.user_id,
            user_name=self.user_name,
            request_id=self.request_id,
            timestamp=self.timestamp,
            channel_id=self.channel_id,
            message_id=self.message_id,
            reply_to=self.reply_to,
            timeout=self.timeout,
            max_steps=self.max_steps,
            streaming=self.streaming,
            response_format=self.response_format,
            metadata=self.metadata.copy(),
            raw_data=self.raw_data.copy(),
            event_callback=self.event_callback,
            status_callback=self.status_callback,
            stream_callback=self.stream_callback,
        )

    def emit_event(self, event_type: SDKEventType, data: Any = None) -> None:
        """Emit an event if callback is registered."""
        if self.event_callback:
            event = SDKEvent(
                type=event_type, data=data, context_id=self.request_id, timestamp=datetime.now()
            )
            self.event_callback(event)


# =============================================================================
# SDK EVENTS
# =============================================================================


@dataclass
class SDKEvent:
    """
    Event emitted during SDK execution.

    Used for:
    - Streaming responses
    - Progress updates
    - Skill execution notifications
    - Error reporting

    Usage:
        client.on("skill_start", lambda e: print(f"Using {e.data['skill']}"))
        client.on("stream", lambda e: print(e.data['delta'], end=""))
    """

    type: SDKEventType
    data: Any = None
    context_id: Optional[str] = None
    seq: int = 0  # Sequence number for ordering guarantees
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "context_id": self.context_id,
            "seq": self.seq,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDKEvent":
        """Create from dictionary."""
        return cls(
            type=SDKEventType(data["type"]),
            data=data.get("data"),
            context_id=data.get("context_id"),
            seq=data.get("seq", 0),
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
            ),
        )


# =============================================================================
# SDK SESSION
# =============================================================================


@dataclass
class SDKSession:
    """
    Persistent session for cross-channel user tracking.

    Features:
    - Persists across restarts (saved to memory layer)
    - Links same user across channels (Telegram + Web = same session)
    - Stores conversation history
    - Tracks user preferences

    Usage:
        session = await session_manager.get_or_create(user_id="user-123")
        session.add_message("user", "Hello")
        await session_manager.save(session)
    """

    session_id: str
    user_id: str

    # Channel associations
    channels: Dict[str, str] = field(default_factory=dict)  # channel_type -> channel_id
    primary_channel: Optional[ChannelType] = None

    # Conversation history
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    # User data
    user_name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message to history, maintaining max size."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )
        # Trim to max_history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]
        self.last_active = datetime.now()
        self.updated_at = datetime.now()

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return self.messages[-limit:]

    def link_channel(self, channel: ChannelType, channel_id: str) -> None:
        """Link a channel to this session."""
        self.channels[channel.value] = channel_id
        if self.primary_channel is None:
            self.primary_channel = channel
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "channels": self.channels,
            "primary_channel": self.primary_channel.value if self.primary_channel else None,
            "messages": self.messages,
            "max_history": self.max_history,
            "user_name": self.user_name,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDKSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            channels=data.get("channels", {}),
            primary_channel=(
                ChannelType(data["primary_channel"]) if data.get("primary_channel") else None
            ),
            messages=data.get("messages", []),
            max_history=data.get("max_history", 50),
            user_name=data.get("user_name"),
            preferences=data.get("preferences", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.now()
            ),
            last_active=(
                datetime.fromisoformat(data["last_active"])
                if "last_active" in data
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# SDK RESPONSE
# =============================================================================


@dataclass
class SDKResponse:
    """
    Standardized response format for SDK.

    Provides consistent structure across:
    - Chat responses
    - Workflow results
    - Skill outputs
    - Streaming chunks

    Usage:
        response = await client.chat("Hello")
        print(response.content)
        print(response.metadata)
    """

    success: bool
    content: Any

    # Response type
    response_format: ResponseFormat = ResponseFormat.TEXT

    # Execution info
    mode: Optional[ExecutionMode] = None
    request_id: Optional[str] = None

    # Timing
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Skills/Agents used
    skills_used: List[str] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    steps_executed: int = 0

    # Errors
    error: Optional[str] = None
    error_code: Optional[str] = None
    errors: List[str] = field(default_factory=list)  # Multiple errors from execution
    stopped_early: bool = False  # True if execution stopped due to failure

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "content": self.content,
            "response_format": self.response_format.value,
            "mode": self.mode.value if self.mode else None,
            "request_id": self.request_id,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "skills_used": self.skills_used,
            "agents_used": self.agents_used,
            "steps_executed": self.steps_executed,
            "error": self.error,
            "error_code": self.error_code,
            "errors": self.errors,
            "stopped_early": self.stopped_early,
            "metadata": self.metadata,
        }

    @classmethod
    def error_response(cls, error: str, error_code: str = "ERROR") -> "SDKResponse":
        """Create an error response."""
        return cls(
            success=False,
            content=None,
            error=error,
            error_code=error_code,
        )


@dataclass
class SDKVoiceResponse(SDKResponse):
    """
    Specialized response for voice interactions (extends SDKResponse).

    Provides voice-specific fields:
    - user_text: Transcribed text from STT
    - audio_data: Generated audio bytes from TTS
    - audio_format: MIME type of audio
    - confidence: STT confidence score
    - provider: Voice provider used (groq, edge, whisper, etc.)

    Usage:
        response = await client.stt(audio_data)
        print(response.user_text)
        print(f"Confidence: {response.confidence}")
    """

    # Voice-specific fields
    user_text: Optional[str] = None  # Transcribed text from STT
    audio_data: Optional[bytes] = None  # Generated audio from TTS
    audio_format: str = "audio/mp3"  # MIME type (audio/mp3, audio/wav, audio/webm)
    confidence: float = 1.0  # STT confidence score (0.0 to 1.0)
    provider: Optional[str] = None  # Provider used (groq, edge, whisper, elevenlabs, local)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update(
            {
                "user_text": self.user_text,
                "audio_data": (
                    self.audio_data.decode("latin1") if self.audio_data else None
                ),  # Base64 encode in real impl
                "audio_format": self.audio_format,
                "confidence": self.confidence,
                "provider": self.provider,
            }
        )
        return base


# =============================================================================
# SDK REQUEST
# =============================================================================


@dataclass
class SDKRequest:
    """
    Standardized request format for SDK.

    Provides consistent structure for:
    - Chat messages
    - Workflow goals
    - Skill invocations
    - Agent commands
    """

    content: str
    mode: ExecutionMode = ExecutionMode.CHAT

    # Optional parameters
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

    # Skill/Agent targeting
    skill_name: Optional[str] = None
    agent_name: Optional[str] = None

    # Streaming
    stream: bool = False

    # Attachments
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "mode": self.mode.value,
            "context": self.context,
            "history": self.history,
            "skill_name": self.skill_name,
            "agent_name": self.agent_name,
            "stream": self.stream,
            "attachments": self.attachments,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDKRequest":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            mode=ExecutionMode(data.get("mode", "chat")),
            context=data.get("context"),
            history=data.get("history"),
            skill_name=data.get("skill_name"),
            agent_name=data.get("agent_name"),
            stream=data.get("stream", False),
            attachments=data.get("attachments", []),
            metadata=data.get("metadata", {}),
        )
