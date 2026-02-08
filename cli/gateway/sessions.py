"""
Persistent Session Manager
==========================

Cross-channel session persistence integrated with memory layer.

Features:
- Sessions persist across restarts (saved to ~/jotty/sessions/)
- Links same user across channels (Telegram + Web = same user)
- Integrates with HierarchicalMemory for context
- Automatic session cleanup for inactive sessions

Usage:
    manager = PersistentSessionManager()

    # Get or create session
    session = await manager.get_or_create(
        user_id="user-123",
        channel=ChannelType.TELEGRAM,
        channel_id="chat-456"
    )

    # Add message
    session.add_message("user", "Hello!")

    # Save session
    await manager.save(session)

    # Find session by channel
    session = await manager.find_by_channel(ChannelType.TELEGRAM, "chat-456")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

# Import SDK types
try:
    from core.foundation.types.sdk_types import (
        ChannelType,
        SDKSession,
        ExecutionContext,
    )
    SDK_TYPES_AVAILABLE = True
except ImportError:
    # Fallback for standalone usage
    SDK_TYPES_AVAILABLE = False
    from enum import Enum
    class ChannelType(Enum):
        CLI = "cli"
        TELEGRAM = "telegram"
        SLACK = "slack"
        DISCORD = "discord"
        WHATSAPP = "whatsapp"
        WEBSOCKET = "websocket"
        HTTP = "http"
        SDK = "sdk"
        WEB = "web"
        CUSTOM = "custom"

    # Minimal SDKSession fallback
    from dataclasses import dataclass, field
    from typing import Dict, List, Any, Optional
    from datetime import datetime

    @dataclass
    class SDKSession:
        session_id: str
        user_id: str
        channels: Dict[str, str] = field(default_factory=dict)
        primary_channel: Optional[ChannelType] = None
        messages: List[Dict[str, Any]] = field(default_factory=list)
        max_history: int = 50
        user_name: Optional[str] = None
        preferences: Dict[str, Any] = field(default_factory=dict)
        created_at: datetime = field(default_factory=datetime.now)
        updated_at: datetime = field(default_factory=datetime.now)
        last_active: datetime = field(default_factory=datetime.now)
        metadata: Dict[str, Any] = field(default_factory=dict)

        def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            if len(self.messages) > self.max_history:
                self.messages = self.messages[-self.max_history:]
            self.last_active = datetime.now()

        def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
            return self.messages[-limit:]

        def link_channel(self, channel: ChannelType, channel_id: str):
            self.channels[channel.value] = channel_id
            if self.primary_channel is None:
                self.primary_channel = channel

        def to_dict(self) -> Dict[str, Any]:
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "channels": self.channels,
                "primary_channel": self.primary_channel.value if self.primary_channel else None,
                "messages": self.messages,
                "user_name": self.user_name,
                "preferences": self.preferences,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "last_active": self.last_active.isoformat(),
                "metadata": self.metadata,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "SDKSession":
            return cls(
                session_id=data["session_id"],
                user_id=data["user_id"],
                channels=data.get("channels", {}),
                primary_channel=ChannelType(data["primary_channel"]) if data.get("primary_channel") else None,
                messages=data.get("messages", []),
                user_name=data.get("user_name"),
                preferences=data.get("preferences", {}),
                created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
                updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
                last_active=datetime.fromisoformat(data["last_active"]) if "last_active" in data else datetime.now(),
                metadata=data.get("metadata", {}),
            )


class PersistentSessionManager:
    """
    Manages persistent sessions across channels.

    Sessions are:
    - Stored in ~/jotty/sessions/{user_id}.json
    - Indexed by channel for fast lookup
    - Automatically cleaned up after inactivity
    - Integrated with memory layer for context
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_inactive_days: int = 30,
        auto_save: bool = True
    ):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory for session storage (default: ~/jotty/sessions)
            max_inactive_days: Days before session is considered stale
            auto_save: Auto-save sessions on modification
        """
        self.storage_dir = storage_dir or (Path.home() / "jotty" / "sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_inactive_days = max_inactive_days
        self.auto_save = auto_save

        # In-memory caches
        self._sessions: Dict[str, SDKSession] = {}  # user_id -> session
        self._channel_index: Dict[str, str] = {}    # channel_key -> user_id

        # Background tasks
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._cleanup_task: Optional[asyncio.Task] = None

    def _get_session_path(self, user_id: str) -> Path:
        """Get file path for a session."""
        # Sanitize user_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.storage_dir / f"{safe_id}.json"

    def _channel_key(self, channel: ChannelType, channel_id: str) -> str:
        """Create a unique key for channel lookup."""
        return f"{channel.value}:{channel_id}"

    async def get_or_create(
        self,
        user_id: str,
        channel: Optional[ChannelType] = None,
        channel_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> SDKSession:
        """
        Get existing session or create new one.

        Args:
            user_id: User identifier
            channel: Optional channel type
            channel_id: Optional channel-specific ID
            user_name: Optional user display name

        Returns:
            SDKSession for the user
        """
        # Check in-memory cache first
        if user_id in self._sessions:
            session = self._sessions[user_id]
            # Link channel if provided
            if channel and channel_id:
                session.link_channel(channel, channel_id)
                self._channel_index[self._channel_key(channel, channel_id)] = user_id
                if self.auto_save:
                    await self.save(session)
            return session

        # Try to load from disk
        session = await self._load_session(user_id)

        if session is None:
            # Create new session
            import uuid
            session = SDKSession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                user_name=user_name
            )
            logger.info(f"Created new session for user {user_id}")

        # Link channel if provided
        if channel and channel_id:
            session.link_channel(channel, channel_id)
            if session.primary_channel is None:
                session.primary_channel = channel
            self._channel_index[self._channel_key(channel, channel_id)] = user_id

        # Cache session
        self._sessions[user_id] = session

        # Update channel index for all linked channels
        for ch_type, ch_id in session.channels.items():
            self._channel_index[f"{ch_type}:{ch_id}"] = user_id

        if self.auto_save:
            await self.save(session)

        return session

    async def find_by_channel(
        self,
        channel: ChannelType,
        channel_id: str
    ) -> Optional[SDKSession]:
        """
        Find session by channel.

        Args:
            channel: Channel type
            channel_id: Channel-specific ID

        Returns:
            SDKSession if found, None otherwise
        """
        key = self._channel_key(channel, channel_id)

        # Check index
        if key in self._channel_index:
            user_id = self._channel_index[key]
            return await self.get_or_create(user_id)

        # Scan all sessions for channel match
        for session_file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                channels = data.get("channels", {})
                if channels.get(channel.value) == channel_id:
                    session = SDKSession.from_dict(data)
                    self._sessions[session.user_id] = session
                    self._channel_index[key] = session.user_id
                    return session
            except Exception as e:
                logger.debug(f"Error reading session file {session_file}: {e}")

        return None

    async def _load_session(self, user_id: str) -> Optional[SDKSession]:
        """Load session from disk."""
        session_path = self._get_session_path(user_id)

        if not session_path.exists():
            return None

        try:
            data = json.loads(session_path.read_text())
            session = SDKSession.from_dict(data)

            # Check if session is stale
            if self._is_stale(session):
                logger.info(f"Session {user_id} is stale, creating new one")
                return None

            return session

        except Exception as e:
            logger.error(f"Error loading session {user_id}: {e}")
            return None

    def _is_stale(self, session: SDKSession) -> bool:
        """Check if session is stale (inactive for too long)."""
        if not session.last_active:
            return False
        cutoff = datetime.now() - timedelta(days=self.max_inactive_days)
        return session.last_active < cutoff

    async def save(self, session: SDKSession) -> None:
        """
        Save session to disk.

        Args:
            session: Session to save
        """
        session.updated_at = datetime.now()
        session_path = self._get_session_path(session.user_id)

        try:
            session_path.write_text(json.dumps(session.to_dict(), indent=2, default=str))
            logger.debug(f"Saved session {session.user_id}")
        except Exception as e:
            logger.error(f"Error saving session {session.user_id}: {e}")

    async def delete(self, user_id: str) -> bool:
        """
        Delete a session.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        session = self._sessions.pop(user_id, None)

        # Remove from channel index
        if session:
            for ch_type, ch_id in session.channels.items():
                key = f"{ch_type}:{ch_id}"
                self._channel_index.pop(key, None)

        # Delete file
        session_path = self._get_session_path(user_id)
        if session_path.exists():
            session_path.unlink()
            return True

        return False

    async def cleanup_stale(self) -> int:
        """
        Clean up stale sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        for session_file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                last_active = data.get("last_active")
                if last_active:
                    last_active_dt = datetime.fromisoformat(last_active)
                    cutoff = datetime.now() - timedelta(days=self.max_inactive_days)
                    if last_active_dt < cutoff:
                        session_file.unlink()
                        cleaned += 1
                        logger.info(f"Cleaned up stale session: {session_file.stem}")
            except Exception as e:
                logger.debug(f"Error checking session {session_file}: {e}")

        return cleaned

    def get_cached(self, user_id: str) -> Optional[SDKSession]:
        """Get session from cache without disk access."""
        return self._sessions.get(user_id)

    def list_active(self) -> List[str]:
        """List all cached session user IDs."""
        return list(self._sessions.keys())

    @property
    def stats(self) -> Dict:
        """Get session manager statistics."""
        return {
            "cached_sessions": len(self._sessions),
            "channel_mappings": len(self._channel_index),
            "storage_dir": str(self.storage_dir),
        }


# Singleton instance
_session_manager: Optional[PersistentSessionManager] = None


def get_session_manager() -> PersistentSessionManager:
    """Get the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = PersistentSessionManager()
    return _session_manager
