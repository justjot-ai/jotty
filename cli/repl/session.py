"""
Session Manager
===============

Manages CLI session state, history, and context.
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Message:
    """A message in the conversation."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """
    Manages CLI session state.

    Features:
    - Conversation history
    - Context window (last N messages for LLM)
    - Session persistence
    - Working directory tracking
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        session_dir: Optional[str] = None,
        context_window: int = 20,
        auto_save: bool = True
    ):
        """
        Initialize session manager.

        Args:
            session_id: Optional session ID (generates if None)
            session_dir: Session storage directory
            context_window: Number of messages in context window
            auto_save: Auto-save on message add
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_dir = Path(session_dir or "~/.jotty/sessions").expanduser()
        self.context_window = context_window
        self.auto_save = auto_save

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
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add message to history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role=role, content=content, metadata=metadata)
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "working_dir": str(self.working_dir),
            "message_count": len(self.conversation_history),
            "context_window": self.context_window,
            "metadata": self.metadata,
        }
