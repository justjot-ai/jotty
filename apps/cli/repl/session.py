"""
Session Manager
===============

Manages CLI session state, history, and context.
Supports cross-interface sync for CLI, Telegram, and Web UI.
"""

import base64
import hashlib
import json
import logging
import secrets
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InterfaceType:
    """Interface type constants for message tracking."""

    CLI = "cli"
    TELEGRAM = "telegram"
    WEB = "web"
    API = "api"


class Message:
    """A message in the conversation with interface tracking and branching support."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        interface: str = InterfaceType.CLI,
        message_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        branch_id: str = "main",
    ) -> None:
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.interface = interface
        self.message_id = message_id or str(uuid.uuid4())[:12]
        self.user_id = user_id
        # Branching support
        self.parent_id = parent_id  # ID of parent message (for tree structure)
        self.branch_id = branch_id  # Branch identifier (main, branch_1, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "interface": self.interface,
            "message_id": self.message_id,
            "user_id": self.user_id,
            "parent_id": self.parent_id,
            "branch_id": self.branch_id,
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
            parent_id=data.get("parent_id"),
            branch_id=data.get("branch_id", "main"),
        )


class ShareLink:
    """Represents a shareable link for a conversation session."""

    def __init__(
        self,
        session_id: str,
        token: Optional[str] = None,
        title: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        access_count: int = 0,
        is_active: bool = True,
        branch_id: str = "main",
    ) -> None:
        self.session_id = session_id
        self.token = token or secrets.token_urlsafe(16)
        self.title = title
        self.expires_at = expires_at
        self.created_at = created_at or datetime.now()
        self.access_count = access_count
        self.is_active = is_active
        self.branch_id = branch_id

    @property
    def is_expired(self) -> bool:
        """Check if the share link has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the share link is active and not expired."""
        return self.is_active and not self.is_expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "token": self.token,
            "title": self.title,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "is_active": self.is_active,
            "branch_id": self.branch_id,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShareLink":
        return cls(
            session_id=data["session_id"],
            token=data.get("token"),
            title=data.get("title"),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            created_at=(
                datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            ),
            access_count=data.get("access_count", 0),
            is_active=data.get("is_active", True),
            branch_id=data.get("branch_id", "main"),
        )


class ShareLinkRegistry:
    """
    Singleton registry for managing share links across sessions.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> Any:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._links: Dict[str, ShareLink] = {}  # token -> ShareLink
                    cls._instance._session_links: Dict[str, List[str]] = (
                        {}
                    )  # session_id -> [tokens]
                    cls._instance._share_dir = Path("~/.jotty/shares").expanduser()
                    cls._instance._share_dir.mkdir(parents=True, exist_ok=True)
                    cls._instance._load_links()
        return cls._instance

    def _load_links(self) -> Any:
        """Load share links from disk."""
        index_file = self._share_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                for link_data in data.get("links", []):
                    link = ShareLink.from_dict(link_data)
                    self._links[link.token] = link
                    if link.session_id not in self._session_links:
                        self._session_links[link.session_id] = []
                    self._session_links[link.session_id].append(link.token)
            except Exception as e:
                logger.error(f"Failed to load share links: {e}")

    def _save_links(self) -> Any:
        """Save share links to disk."""
        index_file = self._share_dir / "index.json"
        try:
            data = {"links": [link.to_dict() for link in self._links.values()]}
            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save share links: {e}")

    def create_link(
        self,
        session_id: str,
        title: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        branch_id: str = "main",
    ) -> ShareLink:
        """Create a new share link for a session."""
        with self._lock:
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)

            link = ShareLink(
                session_id=session_id, title=title, expires_at=expires_at, branch_id=branch_id
            )

            self._links[link.token] = link
            if session_id not in self._session_links:
                self._session_links[session_id] = []
            self._session_links[session_id].append(link.token)

            self._save_links()
            logger.info(f"Created share link {link.token} for session {session_id}")
            return link

    def get_link(self, token: str) -> Optional[ShareLink]:
        """Get a share link by token."""
        link = self._links.get(token)
        if link and link.is_valid:
            return link
        return None

    def get_session_links(self, session_id: str) -> List[ShareLink]:
        """Get all share links for a session."""
        tokens = self._session_links.get(session_id, [])
        return [self._links[t] for t in tokens if t in self._links]

    def revoke_link(self, token: str) -> bool:
        """Revoke a share link."""
        with self._lock:
            if token in self._links:
                self._links[token].is_active = False
                self._save_links()
                logger.info(f"Revoked share link {token}")
                return True
            return False

    def refresh_link(self, token: str, expires_in_days: int = 30) -> Optional[ShareLink]:
        """Refresh a share link with new expiry and token."""
        with self._lock:
            if token not in self._links:
                return None

            old_link = self._links[token]
            new_link = ShareLink(
                session_id=old_link.session_id,
                title=old_link.title,
                expires_at=datetime.now() + timedelta(days=expires_in_days),
                branch_id=old_link.branch_id,
            )

            # Revoke old link
            old_link.is_active = False

            # Add new link
            self._links[new_link.token] = new_link
            if old_link.session_id in self._session_links:
                self._session_links[old_link.session_id].append(new_link.token)

            self._save_links()
            logger.info(f"Refreshed share link {token} -> {new_link.token}")
            return new_link

    def record_access(self, token: str) -> Any:
        """Record an access to a share link."""
        with self._lock:
            if token in self._links:
                self._links[token].access_count += 1
                self._save_links()

    def delete_session_links(self, session_id: str) -> Any:
        """Delete all share links for a session."""
        with self._lock:
            tokens = self._session_links.get(session_id, [])
            for token in tokens:
                if token in self._links:
                    del self._links[token]
            if session_id in self._session_links:
                del self._session_links[session_id]
            self._save_links()


def get_share_link_registry() -> ShareLinkRegistry:
    """Get the global share link registry singleton."""
    return ShareLinkRegistry()


class SessionRegistry:
    """
    Singleton registry for cross-interface session sync.

    Maintains active sessions and allows any interface to
    load/modify shared sessions.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> Any:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: Dict[str, "SessionManager"] = {}
                    cls._instance._session_dir = Path("~/.jotty/sessions").expanduser()
                    cls._instance._session_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def get_session(
        self, session_id: str, create: bool = True, interface: str = InterfaceType.CLI
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

    def remove_session(self, session_id: str) -> Any:
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

    # Constants for temporary chat
    TEMP_CHAT_EXPIRY_DAYS = 30

    def __init__(
        self,
        session_id: Optional[str] = None,
        session_dir: Optional[str] = None,
        context_window: int = 20,
        auto_save: bool = True,
        interface: str = InterfaceType.CLI,
        is_temporary: bool = False,
    ) -> None:
        """
        Initialize session manager.

        Args:
            session_id: Optional session ID (generates if None)
            session_dir: Session storage directory
            context_window: Number of messages in context window
            auto_save: Auto-save on message add
            interface: Source interface (cli, telegram, web)
            is_temporary: Whether this is a temporary (ephemeral) chat
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

        # Temporary chat support
        self.is_temporary = is_temporary
        self.expires_at: Optional[datetime] = None
        if is_temporary:
            self.expires_at = datetime.now() + timedelta(days=self.TEMP_CHAT_EXPIRY_DAYS)
            self.auto_save = False  # Don't persist temporary chats by default

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
        user_id: Optional[str] = None,
    ) -> Any:
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
            user_id=user_id,
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
        messages = self.conversation_history[-self.context_window :]
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

    def clear_history(self) -> Any:
        """Clear conversation history."""
        self.conversation_history.clear()

    def save(self, path: Optional[Path] = None, force: bool = False) -> Any:
        """
        Save session to file.

        Args:
            path: Optional custom path
            force: Force save even for temporary sessions
        """
        # Skip saving temporary sessions unless forced
        if self.is_temporary and not force:
            logger.debug(f"Skipping save for temporary session: {self.session_id}")
            return

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
                "is_temporary": self.is_temporary,
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Session saved: {path}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load(self, session_id: Optional[str] = None, path: Optional[Path] = None) -> Any:
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
            self.created_at = (
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            )
            self.working_dir = Path(data.get("working_dir", "."))
            self.context_window = data.get("context_window", 20)
            self.metadata = data.get("metadata", {})
            self.last_interface = data.get("last_interface", InterfaceType.CLI)
            self.conversation_history = [
                Message.from_dict(m) for m in data.get("conversation_history", [])
            ]

            # Load temporary chat data
            self.is_temporary = data.get("is_temporary", False)
            self.expires_at = (
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            )

            logger.info(f"Session loaded: {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to load session: {e}")

    def list_sessions(
        self, include_temporary: bool = False, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List available sessions.

        Args:
            include_temporary: Include temporary sessions in results
            include_expired: Include expired temporary sessions

        Returns:
            List of session info dicts
        """
        sessions = []

        for file in self.session_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                is_temp = data.get("is_temporary", False)
                expires_at_str = data.get("expires_at")
                expires_at = datetime.fromisoformat(expires_at_str) if expires_at_str else None
                is_expired = is_temp and expires_at and datetime.now() > expires_at

                # Apply filters
                if is_temp and not include_temporary:
                    continue
                if is_expired and not include_expired:
                    continue

                sessions.append(
                    {
                        "session_id": data.get("session_id", file.stem),
                        "created_at": data.get("created_at"),
                        "message_count": len(data.get("conversation_history", [])),
                        "path": str(file),
                        "is_temporary": is_temp,
                        "expires_at": expires_at_str,
                        "is_expired": is_expired,
                    }
                )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> Any:
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
            "branches": self.get_branches(),
            "active_branch": getattr(self, "active_branch", "main"),
            "is_temporary": self.is_temporary,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired,
        }

    @property
    def is_expired(self) -> bool:
        """Check if this temporary session has expired."""
        if not self.is_temporary or not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def set_temporary(self, is_temp: bool, expiry_days: int = None) -> Any:
        """
        Set or unset temporary mode for this session.

        Args:
            is_temp: Whether to make this a temporary session
            expiry_days: Number of days until expiry (default: TEMP_CHAT_EXPIRY_DAYS)
        """
        self.is_temporary = is_temp
        if is_temp:
            days = expiry_days or self.TEMP_CHAT_EXPIRY_DAYS
            self.expires_at = datetime.now() + timedelta(days=days)
        else:
            self.expires_at = None

    @classmethod
    def cleanup_expired_sessions(cls, session_dir: str = None) -> List[str]:
        """
        Remove expired temporary sessions from disk.

        Args:
            session_dir: Session directory to clean up

        Returns:
            List of deleted session IDs
        """
        session_dir = Path(session_dir or "~/.jotty/sessions").expanduser()
        deleted = []

        for file in session_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                if data.get("is_temporary") and data.get("expires_at"):
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    if datetime.now() > expires_at:
                        file.unlink()
                        deleted.append(data.get("session_id", file.stem))
                        logger.info(f"Deleted expired session: {file.stem}")
            except Exception as e:
                logger.error(f"Error checking session {file}: {e}")

        return deleted

    # =========================================================================
    # Conversation Branching Support
    # =========================================================================

    def get_branches(self) -> List[str]:
        """
        Get list of all branch IDs in this session.

        Returns:
            List of unique branch IDs
        """
        branches = set()
        for msg in self.conversation_history:
            branches.add(msg.branch_id)
        return sorted(list(branches))

    def get_branch_messages(self, branch_id: str = "main") -> List[Message]:
        """
        Get messages for a specific branch.

        Args:
            branch_id: Branch identifier

        Returns:
            List of messages in that branch
        """
        return [m for m in self.conversation_history if m.branch_id == branch_id]

    def get_branch_tree(self) -> Dict[str, Any]:
        """
        Get the branch tree structure.

        Returns:
            Dict with branch hierarchy and message counts
        """
        tree = {}
        for msg in self.conversation_history:
            branch = msg.branch_id
            if branch not in tree:
                tree[branch] = {
                    "branch_id": branch,
                    "message_count": 0,
                    "messages": [],
                    "parent_branch": None,
                    "fork_point": None,
                }
            tree[branch]["message_count"] += 1
            tree[branch]["messages"].append(
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "preview": msg.content[:50] + "..." if len(msg.content) > 50 else msg.content,
                    "parent_id": msg.parent_id,
                }
            )

            # Track parent branch from metadata
            if "parent_branch" in msg.metadata:
                tree[branch]["parent_branch"] = msg.metadata["parent_branch"]
            if "fork_point" in msg.metadata:
                tree[branch]["fork_point"] = msg.metadata["fork_point"]

        return tree

    def create_branch(self, from_message_id: str, branch_name: Optional[str] = None) -> str:
        """
        Create a new branch from a specific message.

        Args:
            from_message_id: Message ID to branch from
            branch_name: Optional custom branch name

        Returns:
            New branch ID
        """
        # Find the source message
        source_msg = None
        source_idx = -1
        for i, msg in enumerate(self.conversation_history):
            if msg.message_id == from_message_id:
                source_msg = msg
                source_idx = i
                break

        if not source_msg:
            raise ValueError(f"Message not found: {from_message_id}")

        # Generate branch ID
        existing_branches = self.get_branches()
        if branch_name and branch_name not in existing_branches:
            new_branch_id = branch_name
        else:
            branch_num = 1
            while f"branch_{branch_num}" in existing_branches:
                branch_num += 1
            new_branch_id = f"branch_{branch_num}"

        # Store branch metadata
        if not hasattr(self, "branch_metadata"):
            self.branch_metadata = {}

        self.branch_metadata[new_branch_id] = {
            "parent_branch": source_msg.branch_id,
            "fork_point": from_message_id,
            "created_at": datetime.now().isoformat(),
        }

        # Set active branch
        self.active_branch = new_branch_id

        if self.auto_save:
            self.save()

        logger.info(f"Created branch '{new_branch_id}' from message '{from_message_id}'")
        return new_branch_id

    def edit_message(
        self, message_id: str, new_content: str, create_branch: bool = True
    ) -> Optional[str]:
        """
        Edit a message, optionally creating a new branch.

        Args:
            message_id: ID of message to edit
            new_content: New content for the message
            create_branch: Whether to create a new branch (True) or edit in place (False)

        Returns:
            New branch ID if created, None otherwise
        """
        # Find the message
        target_msg = None
        target_idx = -1
        for i, msg in enumerate(self.conversation_history):
            if msg.message_id == message_id:
                target_msg = msg
                target_idx = i
                break

        if not target_msg:
            raise ValueError(f"Message not found: {message_id}")

        if create_branch:
            # Create a new branch
            new_branch_id = self.create_branch(message_id)

            # Add the edited message to new branch
            edited_msg = Message(
                role=target_msg.role,
                content=new_content,
                interface=target_msg.interface,
                user_id=target_msg.user_id,
                parent_id=target_msg.parent_id,
                branch_id=new_branch_id,
                metadata={
                    **target_msg.metadata,
                    "edited_from": message_id,
                    "parent_branch": target_msg.branch_id,
                    "fork_point": message_id,
                },
            )
            self.conversation_history.append(edited_msg)

            if self.auto_save:
                self.save()

            return new_branch_id
        else:
            # Edit in place
            target_msg.content = new_content
            target_msg.metadata["edited_at"] = datetime.now().isoformat()

            if self.auto_save:
                self.save()

            return None

    def switch_branch(self, branch_id: str) -> bool:
        """
        Switch to a different branch.

        Args:
            branch_id: Branch to switch to

        Returns:
            True if successful
        """
        if branch_id not in self.get_branches():
            raise ValueError(f"Branch not found: {branch_id}")

        self.active_branch = branch_id
        logger.info(f"Switched to branch '{branch_id}'")

        if self.auto_save:
            self.save()

        return True

    def get_context_for_branch(self, branch_id: str = None) -> List[Dict[str, str]]:
        """
        Get context window for a specific branch.

        Follows the branch lineage back to main if needed.

        Args:
            branch_id: Branch to get context for (default: active branch)

        Returns:
            List of messages as dicts for LLM context
        """
        branch_id = branch_id or getattr(self, "active_branch", "main")

        # Get messages for this branch
        branch_messages = self.get_branch_messages(branch_id)

        # If branch has parent, include parent messages up to fork point
        if hasattr(self, "branch_metadata") and branch_id in self.branch_metadata:
            meta = self.branch_metadata[branch_id]
            parent_branch = meta.get("parent_branch")
            fork_point = meta.get("fork_point")

            if parent_branch and fork_point:
                parent_messages = []
                for msg in self.get_branch_messages(parent_branch):
                    parent_messages.append(msg)
                    if msg.message_id == fork_point:
                        break
                branch_messages = parent_messages + branch_messages

        # Apply context window limit
        messages = branch_messages[-self.context_window :]
        return [{"role": m.role, "content": m.content} for m in messages]

    def delete_branch(self, branch_id: str) -> bool:
        """
        Delete a branch and all its messages.

        Cannot delete 'main' branch.

        Args:
            branch_id: Branch to delete

        Returns:
            True if successful
        """
        if branch_id == "main":
            raise ValueError("Cannot delete main branch")

        # Remove all messages in this branch
        self.conversation_history = [
            m for m in self.conversation_history if m.branch_id != branch_id
        ]

        # Remove branch metadata
        if hasattr(self, "branch_metadata") and branch_id in self.branch_metadata:
            del self.branch_metadata[branch_id]

        # Switch to main if we deleted active branch
        if getattr(self, "active_branch", "main") == branch_id:
            self.active_branch = "main"

        if self.auto_save:
            self.save()

        logger.info(f"Deleted branch '{branch_id}'")
        return True
