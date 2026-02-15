"""
Channel Router
==============

Routes messages between external channels (Telegram, Slack, Discord)
and Jotty agents via ModeRouter.

Integrates with:
- ModeRouter for unified request processing (not JottyCLI)
- PersistentSessionManager for cross-channel session persistence
- ExecutionContext for unified context passing
- ChannelResponderRegistry for registry-based responses
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Absolute imports - single source of truth
from Jotty.sdk import ChannelType, ExecutionContext, ExecutionMode


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
            "attachments": self.attachments,
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
    - Persistent session management (survives restarts)
    - Cross-channel user linking
    - ExecutionContext integration
    - Agent routing based on channel/user
    - Message queue for async processing
    """

    def __init__(self, use_persistent_sessions: bool = True) -> None:
        self._handlers: Dict[ChannelType, Callable] = {}
        self._responders: Dict[ChannelType, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._cli = None
        self._running = False
        self._trust_manager = None

        # Use persistent sessions by default
        self._use_persistent_sessions = use_persistent_sessions
        self._session_manager = None
        self._legacy_sessions: Dict[str, Dict[str, Any]] = {}  # Fallback

    def _get_session_manager(self) -> Any:
        """Get persistent session manager (lazy init)."""
        if self._session_manager is None and self._use_persistent_sessions:
            try:
                from .sessions import get_session_manager

                self._session_manager = get_session_manager()
            except ImportError:
                logger.warning("Persistent sessions not available, using legacy")
                self._use_persistent_sessions = False
        return self._session_manager

    def set_trust_manager(self, trust_manager: Any) -> Any:
        """Set trust manager for message authorization."""
        self._trust_manager = trust_manager
        logger.info("Trust manager configured")

    def set_cli(self, cli: Any) -> Any:
        """Set JottyCLI instance for processing messages."""
        self._cli = cli

    def register_handler(self, channel: ChannelType, handler: Callable) -> Any:
        """Register incoming message handler for a channel."""
        self._handlers[channel] = handler
        logger.info(f"Registered handler for {channel.value}")

    def register_responder(self, channel: ChannelType, responder: Callable) -> Any:
        """Register response sender for a channel."""
        self._responders[channel] = responder
        logger.info(f"Registered responder for {channel.value}")

    async def handle_message(self, event: MessageEvent) -> Any:
        """Handle incoming message from any channel."""
        logger.info(f"[{event.channel.value}] {event.user_name}: {event.content[:50]}...")

        # Check trust if trust manager is configured
        if self._trust_manager:
            trust_result = self._trust_manager.check_message(
                event.channel, event.user_id, event.content
            )

            if not trust_result.get("proceed"):
                # User not authorized - send trust response
                response_text = trust_result.get("response", "Not authorized")
                await self._send_response(
                    ResponseEvent(
                        channel=event.channel,
                        channel_id=event.channel_id,
                        content=response_text,
                        reply_to=event.message_id,
                    )
                )
                return response_text

            # Check if pairing just succeeded - send success message
            if (
                trust_result.get("response")
                and "successful" in trust_result.get("response", "").lower()
            ):
                await self._send_response(
                    ResponseEvent(
                        channel=event.channel,
                        channel_id=event.channel_id,
                        content=trust_result["response"],
                        reply_to=event.message_id,
                    )
                )
                # Continue processing the original message if it wasn't just the code
                if event.content.strip().isdigit() and len(event.content.strip()) == 6:
                    return trust_result["response"]

        # Get or create session (persistent or legacy)
        session_manager = self._get_session_manager()
        session_data = None
        sdk_session = None

        if session_manager:
            # Use persistent sessions
            try:
                # Map local ChannelType to SDK ChannelType
                sdk_channel = SDKChannelType(event.channel.value) if SDK_TYPES_AVAILABLE else None
                sdk_session = await session_manager.get_or_create(
                    user_id=event.user_id,
                    channel=sdk_channel,
                    channel_id=event.channel_id,
                    user_name=event.user_name,
                )
                # Add message to session history
                sdk_session.add_message(
                    "user",
                    event.content,
                    {"channel": event.channel.value, "message_id": event.message_id},
                )
                session_data = {"context": sdk_session.get_history(10)}
            except Exception as e:
                logger.warning(f"Persistent session error: {e}, using legacy")

        if session_data is None:
            # Fallback to legacy in-memory sessions
            session_key = f"{event.channel.value}:{event.channel_id}:{event.user_id}"
            if session_key not in self._legacy_sessions:
                self._legacy_sessions[session_key] = {
                    "created": datetime.now().isoformat(),
                    "message_count": 0,
                    "context": [],
                }

            session_data = self._legacy_sessions[session_key]
            session_data["message_count"] = session_data.get("message_count", 0) + 1
            session_data["last_message"] = datetime.now().isoformat()

            # Add to context (keep last 10 messages)
            session_data["context"].append(
                {"role": "user", "content": event.content, "timestamp": event.timestamp.isoformat()}
            )
            session_data["context"] = session_data["context"][-10:]

        # Create ExecutionContext if SDK types available
        exec_context = None
        if SDK_TYPES_AVAILABLE:
            try:
                exec_context = ExecutionContext(
                    mode=ExecutionMode.CHAT,
                    channel=SDKChannelType(event.channel.value),
                    session_id=sdk_session.session_id if sdk_session else event.session_id,
                    user_id=event.user_id,
                    user_name=event.user_name,
                    channel_id=event.channel_id,
                    message_id=event.message_id,
                    reply_to=event.reply_to,
                    raw_data=event.raw_data,
                )
            except Exception as e:
                logger.debug(f"Could not create ExecutionContext: {e}")

        # Process with Jotty
        response_text = await self._process_with_jotty(event, session_data, exec_context)

        # Add response to session
        if sdk_session:
            sdk_session.add_message("assistant", response_text)
            try:
                await session_manager.save(sdk_session)
            except Exception as e:
                logger.warning(f"Failed to save session: {e}")
        elif session_data:
            session_data["context"].append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Send response back
        await self._send_response(
            ResponseEvent(
                channel=event.channel,
                channel_id=event.channel_id,
                content=response_text,
                reply_to=event.message_id,
            )
        )

        return response_text

    async def _process_with_jotty(
        self, event: MessageEvent, session: Dict, context: Optional[Any] = None
    ) -> str:
        """
        Process message via ModeRouter (preferred) or JottyCLI (fallback).

        ModeRouter is the canonical execution path - routes directly to
        AutoAgent without going through CLI abstractions.

        Args:
            event: The incoming message event
            session: Session data with history
            context: Optional ExecutionContext for enhanced processing
        """
        # ModeRouter: canonical execution path
        try:
            from Jotty.core.interface.api.mode_router import get_mode_router

            router = get_mode_router()

            # Create ExecutionContext if not provided
            if context is None:
                context = ExecutionContext(
                    mode=ExecutionMode.CHAT,
                    channel=event.channel,
                    session_id=event.session_id or "gateway",
                    user_id=event.user_id,
                    user_name=event.user_name,
                    channel_id=event.channel_id,
                    message_id=event.message_id,
                )

            # Add conversation history to context
            history = session.get("context", [])
            if history:
                context.metadata["conversation_history"] = history[-6:]

            result = await router.chat(event.content, context)

            if result.success and result.content:
                return str(result.content)
            elif result.error:
                return f"Error: {result.error}"
            else:
                return str(result.content or "Task completed")

        except Exception as e:
            logger.warning(f"ModeRouter failed, falling back to CLI: {e}")

        # Fallback: JottyCLI
        try:
            if self._cli:
                if context and hasattr(self._cli, "run_once_with_context"):
                    result = await self._cli.run_once_with_context(
                        event.content, context=context, history=session.get("context", [])
                    )
                else:
                    result = await self._cli.run_once(event.content)

                if hasattr(result, "output"):
                    return result.output or str(result)
                return str(result)
            else:
                return f"Received: {event.content}"
        except Exception as e:
            logger.error(f"Jotty processing error: {e}", exc_info=True)
            return f"Error processing message: {str(e)}"

    async def _send_response(self, response: ResponseEvent) -> Any:
        """Send response to the appropriate channel with formatting."""
        # Apply channel-specific formatting before sending
        try:
            from .responders import get_responder_registry

            registry = get_responder_registry()
            response.content = registry.format_for_channel(response.content, response.channel)
        except Exception:
            pass  # Send unformatted if formatting fails

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
        # Try persistent session manager first
        session_manager = self._get_session_manager()
        if session_manager:
            session = session_manager.get_cached(user_id)
            if session:
                return {"context": session.get_history(10)}

        # Fallback to legacy
        session_key = f"{channel.value}:{channel_id}:{user_id}"
        return self._legacy_sessions.get(session_key)

    async def get_session_async(
        self, channel: ChannelType, channel_id: str, user_id: str
    ) -> Optional[Dict]:
        """Get session for a user/channel (async version for persistent lookup)."""
        session_manager = self._get_session_manager()
        if session_manager:
            try:
                sdk_channel = SDKChannelType(channel.value) if SDK_TYPES_AVAILABLE else None
                session = await session_manager.find_by_channel(sdk_channel, channel_id)
                if session:
                    return {"context": session.get_history(10), "session": session}
            except Exception as e:
                logger.debug(f"Async session lookup failed: {e}")

        # Fallback to sync method
        return self.get_session(channel, channel_id, user_id)

    def clear_session(self, channel: ChannelType, channel_id: str, user_id: str) -> Any:
        """Clear session for a user/channel."""
        # Clear from persistent manager
        session_manager = self._get_session_manager()
        if session_manager:
            # Schedule async deletion
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(session_manager.delete(user_id))
            except RuntimeError:
                pass  # No running loop — skip async cleanup

        # Clear from legacy
        session_key = f"{channel.value}:{channel_id}:{user_id}"
        if session_key in self._legacy_sessions:
            del self._legacy_sessions[session_key]

    @property
    def active_sessions(self) -> int:
        """Get count of active sessions."""
        session_manager = self._get_session_manager()
        if session_manager:
            return len(session_manager.list_active()) + len(self._legacy_sessions)
        return len(self._legacy_sessions)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = {
            "active_sessions": self.active_sessions,
            "handlers": [h.value for h in self._handlers.keys()],
            "responders": [r.value for r in self._responders.keys()],
            "persistent_sessions_enabled": self._use_persistent_sessions,
        }

        session_manager = self._get_session_manager()
        if session_manager:
            stats["session_manager"] = session_manager.stats

        return stats
