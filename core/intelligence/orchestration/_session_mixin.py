"""
SwarmIntelligence Session Mixin
================================

Extracted from swarm_intelligence.py — handles isolated agent sessions
(moltbot pattern for per-context session isolation).
"""

import hashlib
import time
import logging
from typing import Dict, List, Optional

from .swarm_data_structures import AgentSession

logger = logging.getLogger(__name__)


class SessionMixin:
    """Mixin for agent session management."""

    def create_session(self, agent_name: str, context: str = "main") -> str:
        """Create isolated session for an agent."""
        session_id = hashlib.md5(f"{agent_name}:{context}:{time.time()}".encode()).hexdigest()[:12]
        self.sessions[session_id] = AgentSession(
            session_id=session_id,
            agent_name=agent_name,
            context=context
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def session_send(self, session_id: str, from_agent: str, content: str, metadata: Dict = None) -> bool:
        """Send message to a session (moltbot sessions_send pattern)."""
        session = self.sessions.get(session_id)
        if session:
            session.add_message(from_agent, content, metadata)
            return True
        return False

    def session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get session history (moltbot sessions_history pattern)."""
        session = self.sessions.get(session_id)
        if session:
            return session.messages[-limit:]
        return []

    def sessions_list(self, agent_name: str = None) -> List[Dict]:
        """List sessions (moltbot sessions_list pattern)."""
        sessions = []
        for sid, session in self.sessions.items():
            if agent_name is None or session.agent_name == agent_name:
                sessions.append({
                    'session_id': sid,
                    'agent': session.agent_name,
                    'context': session.context,
                    'message_count': len(session.messages),
                    'last_active': session.last_active
                })
        return sessions
