"""
Channel Router
==============

Routes messages between external channels (Telegram, Slack, Discord)
and Jotty agents.
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Supported channel types."""
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    WEBSOCKET = "websocket"
    HTTP = "http"


@dataclass
class MessageEvent:
    """Incoming message from any channel."""
    channel: ChannelType
    channel_id: str  # Chat/channel ID
    user_id: str
    user_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Channel-specific data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    # Routing metadata
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "attachments": self.attachments
        }


@dataclass
class ResponseEvent:
    """Outgoing response to a channel."""
    channel: ChannelType
    channel_id: str
    content: str
    reply_to: Optional[str] = None
    attachments: List[str] = field(default_factory=list)


class ChannelRouter:
    """
    Routes messages between channels and Jotty agents.

    Features:
    - Multi-channel support (Telegram, Slack, Discord, WhatsApp)
    - Session management per user/channel
    - Agent routing based on channel/user
    - Message queue for async processing
    """

    def __init__(self):
        self._handlers: Dict[ChannelType, Callable] = {}
        self._responders: Dict[ChannelType, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._sessions: Dict[str, Dict[str, Any]] = {}  # channel:user -> session
        self._cli = None
        self._running = False
        self._trust_manager = None

    def set_trust_manager(self, trust_manager):
        """Set trust manager for message authorization."""
        self._trust_manager = trust_manager
        logger.info("Trust manager configured")

    def set_cli(self, cli):
        """Set JottyCLI instance for processing messages."""
        self._cli = cli

    def register_handler(self, channel: ChannelType, handler: Callable):
        """Register incoming message handler for a channel."""
        self._handlers[channel] = handler
        logger.info(f"Registered handler for {channel.value}")

    def register_responder(self, channel: ChannelType, responder: Callable):
        """Register response sender for a channel."""
        self._responders[channel] = responder
        logger.info(f"Registered responder for {channel.value}")

    async def handle_message(self, event: MessageEvent):
        """Handle incoming message from any channel."""
        logger.info(f"[{event.channel.value}] {event.user_name}: {event.content[:50]}...")

        # Check trust if trust manager is configured
        if self._trust_manager:
            trust_result = self._trust_manager.check_message(
                event.channel,
                event.user_id,
                event.content
            )

            if not trust_result.get("proceed"):
                # User not authorized - send trust response
                response_text = trust_result.get("response", "Not authorized")
                await self._send_response(ResponseEvent(
                    channel=event.channel,
                    channel_id=event.channel_id,
                    content=response_text,
                    reply_to=event.message_id
                ))
                return response_text

            # Check if pairing just succeeded - send success message
            if trust_result.get("response") and "successful" in trust_result.get("response", "").lower():
                await self._send_response(ResponseEvent(
                    channel=event.channel,
                    channel_id=event.channel_id,
                    content=trust_result["response"],
                    reply_to=event.message_id
                ))
                # Continue processing the original message if it wasn't just the code
                if event.content.strip().isdigit() and len(event.content.strip()) == 6:
                    return trust_result["response"]

        # Get or create session
        session_key = f"{event.channel.value}:{event.channel_id}:{event.user_id}"
        if session_key not in self._sessions:
            self._sessions[session_key] = {
                "created": datetime.now().isoformat(),
                "message_count": 0,
                "context": []
            }

        session = self._sessions[session_key]
        session["message_count"] += 1
        session["last_message"] = datetime.now().isoformat()

        # Add to context (keep last 10 messages)
        session["context"].append({
            "role": "user",
            "content": event.content,
            "timestamp": event.timestamp.isoformat()
        })
        session["context"] = session["context"][-10:]

        # Process with Jotty
        response_text = await self._process_with_jotty(event, session)

        # Add response to context
        session["context"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })

        # Send response back
        await self._send_response(ResponseEvent(
            channel=event.channel,
            channel_id=event.channel_id,
            content=response_text,
            reply_to=event.message_id
        ))

        return response_text

    async def _process_with_jotty(self, event: MessageEvent, session: Dict) -> str:
        """Process message with Jotty CLI."""
        try:
            if self._cli:
                # Run message through Jotty
                result = await self._cli.run_once(event.content)
                if hasattr(result, 'output'):
                    return result.output or str(result)
                return str(result)
            else:
                return f"Received: {event.content}"
        except Exception as e:
            logger.error(f"Jotty processing error: {e}", exc_info=True)
            return f"Error processing message: {str(e)}"

    async def _send_response(self, response: ResponseEvent):
        """Send response to the appropriate channel."""
        responder = self._responders.get(response.channel)
        if responder:
            try:
                await responder(response)
                logger.info(f"[{response.channel.value}] Sent response to {response.channel_id}")
            except Exception as e:
                logger.error(f"Response error: {e}", exc_info=True)
        else:
            logger.warning(f"No responder for {response.channel.value}")

    def get_session(self, channel: ChannelType, channel_id: str, user_id: str) -> Optional[Dict]:
        """Get session for a user/channel."""
        session_key = f"{channel.value}:{channel_id}:{user_id}"
        return self._sessions.get(session_key)

    def clear_session(self, channel: ChannelType, channel_id: str, user_id: str):
        """Clear session for a user/channel."""
        session_key = f"{channel.value}:{channel_id}:{user_id}"
        if session_key in self._sessions:
            del self._sessions[session_key]

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "active_sessions": self.active_sessions,
            "handlers": list(self._handlers.keys()),
            "responders": list(self._responders.keys())
        }
