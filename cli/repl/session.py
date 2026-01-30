"""
Session Manager
===============

Manages CLI session state, history, and context.
Supports cross-interface sync for CLI, Telegram, and Web UI.
"""

import json
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class InterfaceType:
    """Interface type constants for message tracking."""
    CLI = "cli"
    TELEGRAM = "telegram"
    WEB = "web"
    API = "api"


class Message:
    """A message in the conversation with interface tracking."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        interface: str = InterfaceType.CLI,
        message_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.interface = interface
        self.message_id = message_id or str(uuid.uuid4())[:12]
        self.user_id = user_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "interface": self.interface,
            "message_id": self.message_id,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            metadata=data.get("metadata", {}),
            interface=data.get("interface", InterfaceType.CLI),
            message_id=data.get("message_id"),
            user_id=data.get("user_id"),
        )


class SessionRegistry:
    """
    Singleton registry for cross-interface session sync.

    Maintains active sessions and allows any interface to
    load/modify shared sessions.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: Dict[str, "SessionManager"] = {}
                    cls._instance._session_dir = Path("~/.jotty/sessions").expanduser()
                    cls._instance._session_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def get_session(
        self,
        session_id: str,
        create: bool = True,
        interface: str = InterfaceType.CLI
    ) -> Optional["SessionManager"]:
        """
        Get or create a session by ID.

        Args:
            session_id: Session identifier
            create: Create if not exists
            interface: Interface requesting the session

        Returns:
            SessionManager instance or None
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_interface = interface
                return session

            # Try loading from disk
            session_file = self._session_dir / f"{session_id}.json"
            if session_file.exists():
                session = SessionManager(session_id=session_id)
                session.load()
                session.last_interface = interface
                self._sessions[session_id] = session
                return session

            # Create new if requested
            if create:
                session = SessionManager(session_id=session_id)
                session.last_interface = interface
                self._sessions[session_id] = session
                return session

            return None

    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())

    def remove_session(self, session_id: str):
        """Remove session from registry (doesn't delete from disk)."""
        with self._lock:
            self._sessions.pop(session_id, None)


def get_session_registry() -> SessionRegistry:
    """Get the global session registry singleton."""
    return SessionRegistry()


class SessionManager:
    """
    Manages CLI session state.

    Features:
    - Conversation history with interface tracking
    - Context window (last N messages for LLM)
    - Session persistence
    - Working directory tracking
    - Cross-interface sync via SessionRegistry
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        session_dir: Optional[str] = None,
        context_window: int = 20,
        auto_save: bool = True,
        interface: str = InterfaceType.CLI
    ):
        """
        Initialize session manager.

        Args:
            session_id: Optional session ID (generates if None)
            session_dir: Session storage directory
            context_window: Number of messages in context window
            auto_save: Auto-save on message add
            interface: Source interface (cli, telegram, web)
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_dir = Path(session_dir or "~/.jotty/sessions").expanduser()
        self.context_window = context_window
        self.auto_save = auto_save
        self.last_interface = interface

        self.conversation_history: List[Message] = []
        self.working_dir = Path.cwd()
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}

        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_file(self) -> Path:
        """Get session file path."""
        return self.session_dir / f"{self.session_id}.json"

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        interface: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Add message to history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
            interface: Source interface (defaults to last_interface)
            user_id: Optional user identifier
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
            interface=interface or self.last_interface,
            user_id=user_id
        )
        self.conversation_history.append(message)

        if self.auto_save:
            self.save()

    def get_context(self) -> List[Dict[str, str]]:
        """
        Get context window for LLM.

        Returns:
            List of last N messages as dicts
        """
        messages = self.conversation_history[-self.context_window:]
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            limit: Max messages to return

        Returns:
            List of message dicts
        """
        messages = self.conversation_history
        if limit:
            messages = messages[-limit:]
        return [m.to_dict() for m in messages]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()

    def save(self, path: Optional[Path] = None):
        """
        Save session to file.

        Args:
            path: Optional custom path
        """
        path = path or self.session_file

        try:
            data = {
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "working_dir": str(self.working_dir),
                "context_window": self.context_window,
                "metadata": self.metadata,
                "last_interface": self.last_interface,
                "conversation_history": [m.to_dict() for m in self.conversation_history],
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Session saved: {path}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load(self, session_id: Optional[str] = None, path: Optional[Path] = None):
        """
        Load session from file.

        Args:
            session_id: Session ID to load
            path: Optional custom path
        """
        if session_id:
            path = self.session_dir / f"{session_id}.json"
        path = path or self.session_file

        if not path.exists():
            logger.warning(f"Session file not found: {path}")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.session_id = data.get("session_id", self.session_id)
            self.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
            self.working_dir = Path(data.get("working_dir", "."))
            self.context_window = data.get("context_window", 20)
            self.metadata = data.get("metadata", {})
            self.last_interface = data.get("last_interface", InterfaceType.CLI)
            self.conversation_history = [
                Message.from_dict(m) for m in data.get("conversation_history", [])
            ]

            logger.info(f"Session loaded: {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to load session: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available sessions.

        Returns:
            List of session info dicts
        """
        sessions = []

        for file in self.session_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", file.stem),
                    "created_at": data.get("created_at"),
                    "message_count": len(data.get("conversation_history", [])),
                    "path": str(file),
                })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_session(self, session_id: str):
        """
        Delete a session.

        Args:
            session_id: Session ID to delete
        """
        path = self.session_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Session deleted: {session_id}")

    def get_messages_by_interface(self, interface: str) -> List[Message]:
        """
        Get messages from a specific interface.

        Args:
            interface: Interface type (cli, telegram, web)

        Returns:
            List of messages from that interface
        """
        return [m for m in self.conversation_history if m.interface == interface]

    def get_interface_summary(self) -> Dict[str, int]:
        """
        Get message count by interface.

        Returns:
            Dict mapping interface type to message count
        """
        summary: Dict[str, int] = {}
        for msg in self.conversation_history:
            iface = msg.interface
            summary[iface] = summary.get(iface, 0) + 1
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "working_dir": str(self.working_dir),
            "message_count": len(self.conversation_history),
            "context_window": self.context_window,
            "metadata": self.metadata,
            "last_interface": self.last_interface,
            "interface_summary": self.get_interface_summary(),
        }
