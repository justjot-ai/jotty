"""
WebSocket Handler
=================

WebSocket support for streaming responses to Web UI.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""
    websocket: Any  # WebSocket object
    session_id: str
    user_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    _id: str = field(default_factory=lambda: str(id(object())))

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, WebSocketConnection):
            return self._id == other._id
        return False


class WebSocketManager:
    """
    Manages WebSocket connections for real-time streaming.

    Features:
    - Connection tracking by session
    - Broadcast to session
    - Stream tokens as they're generated
    """

    def __init__(self):
        self._connections: Dict[str, Set[WebSocketConnection]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: Any,
        session_id: str,
        user_id: str = "anonymous"
    ) -> WebSocketConnection:
        """
        Register a new WebSocket connection.

        Args:
            websocket: WebSocket object
            session_id: Session to join
            user_id: User identifier

        Returns:
            WebSocketConnection object
        """
        conn = WebSocketConnection(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id
        )

        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = set()
            self._connections[session_id].add(conn)

        logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
        return conn

    async def disconnect(self, conn: WebSocketConnection):
        """
        Remove a WebSocket connection.

        Args:
            conn: Connection to remove
        """
        async with self._lock:
            if conn.session_id in self._connections:
                self._connections[conn.session_id].discard(conn)
                if not self._connections[conn.session_id]:
                    del self._connections[conn.session_id]

        logger.info(f"WebSocket disconnected: session={conn.session_id}")

    async def send_to_session(
        self,
        session_id: str,
        message: Dict[str, Any]
    ):
        """
        Send message to all connections in a session.

        Args:
            session_id: Target session
            message: Message dict to send
        """
        async with self._lock:
            connections = self._connections.get(session_id, set()).copy()

        dead_connections = []

        for conn in connections:
            try:
                await conn.websocket.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket: {e}")
                dead_connections.append(conn)

        # Clean up dead connections
        for conn in dead_connections:
            await self.disconnect(conn)

    async def stream_chunk(
        self,
        session_id: str,
        chunk: str,
        message_id: Optional[str] = None
    ):
        """
        Stream a content chunk to session.

        Args:
            session_id: Target session
            chunk: Content chunk
            message_id: Optional message ID
        """
        await self.send_to_session(session_id, {
            "type": "stream",
            "chunk": chunk,
            "message_id": message_id,
            "timestamp": datetime.now().isoformat()
        })

    async def send_status(
        self,
        session_id: str,
        stage: str,
        detail: str = ""
    ):
        """
        Send status update to session.

        Args:
            session_id: Target session
            stage: Status stage
            detail: Status detail
        """
        await self.send_to_session(session_id, {
            "type": "status",
            "stage": stage,
            "detail": detail,
            "timestamp": datetime.now().isoformat()
        })

    async def send_complete(
        self,
        session_id: str,
        message_id: str,
        content: str,
        output_path: Optional[str] = None
    ):
        """
        Send completion message to session.

        Args:
            session_id: Target session
            message_id: Message ID
            content: Full content
            output_path: Optional output file path
        """
        await self.send_to_session(session_id, {
            "type": "complete",
            "message_id": message_id,
            "content": content,
            "output_path": output_path,
            "timestamp": datetime.now().isoformat()
        })

    async def send_error(
        self,
        session_id: str,
        error: str
    ):
        """
        Send error to session.

        Args:
            session_id: Target session
            error: Error message
        """
        await self.send_to_session(session_id, {
            "type": "error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_session_connection_count(self, session_id: str) -> int:
        """Get number of connections in a session."""
        return len(self._connections.get(session_id, set()))

    def get_all_sessions(self) -> list:
        """Get all active session IDs."""
        return list(self._connections.keys())


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
