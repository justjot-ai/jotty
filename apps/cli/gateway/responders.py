"""
Channel Responder Registry
==========================

Registry-based channel responders replacing hardcoded skill imports.
Each channel type has a responder that sends messages back through
the appropriate channel (Telegram, Slack, Discord, etc.).

This module:
- Discovers responder skills from the registry
- Provides fallback responders when skills aren't available
- Handles channel-specific formatting
- Integrates with ExecutionContext for full context awareness
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Absolute imports - single source of truth
from Jotty.sdk import ChannelType, ExecutionContext


@dataclass
class ResponseEvent:
    """Outgoing response to a channel."""

    channel: ChannelType
    channel_id: str
    content: str
    reply_to: Optional[str] = None
    attachments: list = None
    context: Optional[ExecutionContext] = None

    def __post_init__(self) -> None:
        if self.attachments is None:
            self.attachments = []


class ChannelResponderRegistry:
    """
    Registry for channel responders.

    Replaces hardcoded skill imports with registry-based discovery.
    Provides fallback responders and handles channel-specific formatting.

    Usage:
        registry = ChannelResponderRegistry()
        await registry.send(ResponseEvent(
            channel=ChannelType.TELEGRAM,
            channel_id="123456",
            content="Hello!"
        ))
    """

    def __init__(self) -> None:
        self._responders: Dict[ChannelType, Callable] = {}
        self._skill_cache: Dict[str, Any] = {}
        self._initialized = False

    def _discover_skill(self, skill_name: str) -> Optional[Callable]:
        """
        Discover a skill tool from the registry.

        Returns the tool function if found, None otherwise.
        """
        if skill_name in self._skill_cache:
            return self._skill_cache[skill_name]

        try:
            # Try to import from skills dynamically
            # This avoids hardcoded imports
            from Jotty.core.capabilities.registry.unified_registry import get_unified_registry

            registry = get_unified_registry()

            # Get the skill
            skill = registry.get_skill(skill_name)
            if skill and hasattr(skill, "tools"):
                # Get the primary send tool
                for tool in skill.tools:
                    if "send" in tool.name.lower() or "message" in tool.name.lower():
                        self._skill_cache[skill_name] = tool.function
                        return tool.function

            # Fallback: try direct import
            return self._try_direct_import(skill_name)

        except Exception as e:
            logger.debug(f"Could not discover skill {skill_name}: {e}")
            return self._try_direct_import(skill_name)

    def _try_direct_import(self, skill_name: str) -> Optional[Callable]:
        """Try direct import as fallback."""
        import_map = {
            "telegram-sender": ("skills.telegram_sender.tools", "send_telegram_message_tool"),
            "slack": ("skills.slack.tools", "send_message_tool"),
            "discord": ("skills.discord.tools", "send_message_tool"),
            "whatsapp": ("skills.whatsapp.tools", "send_whatsapp_message_tool"),
        }

        if skill_name in import_map:
            module_path, func_name = import_map[skill_name]
            try:
                import importlib

                module = importlib.import_module(module_path)
                func = getattr(module, func_name, None)
                if func:
                    self._skill_cache[skill_name] = func
                    return func
            except Exception as e:
                logger.debug(f"Direct import failed for {skill_name}: {e}")

        return None

    def _initialize_responders(self) -> Any:
        """Initialize responders for all channels."""
        if self._initialized:
            return

        # Telegram responder
        async def telegram_responder(response: ResponseEvent) -> Any:
            tool = self._discover_skill("telegram-sender")
            if tool:
                result = tool({"chat_id": response.channel_id, "message": response.content})
                # Handle both sync and async tools
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning("Telegram sender skill not available")

        # Slack responder
        async def slack_responder(response: ResponseEvent) -> Any:
            tool = self._discover_skill("slack")
            if tool:
                params = {
                    "channel": response.channel_id,
                    "text": response.content,
                }
                if response.reply_to:
                    params["thread_ts"] = response.reply_to

                result = tool(params)
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning("Slack sender skill not available")

        # Discord responder
        async def discord_responder(response: ResponseEvent) -> Any:
            tool = self._discover_skill("discord")
            if tool:
                result = tool({"channel_id": response.channel_id, "content": response.content})
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning("Discord sender skill not available")

        # WhatsApp responder
        async def whatsapp_responder(response: ResponseEvent) -> Any:
            tool = self._discover_skill("whatsapp")
            if tool:
                result = tool({"to": response.channel_id, "message": response.content})
                if hasattr(result, "__await__"):
                    await result
            else:
                logger.warning("WhatsApp sender skill not available")

        # HTTP responder (no-op, response returned directly)
        async def http_responder(response: ResponseEvent) -> Any:
            # HTTP responses are returned directly, not sent via skill
            pass

        # WebSocket responder (handled by gateway server)
        async def websocket_responder(response: ResponseEvent) -> Any:
            # WebSocket responses are handled by the gateway server directly
            pass

        self._responders = {
            ChannelType.TELEGRAM: telegram_responder,
            ChannelType.SLACK: slack_responder,
            ChannelType.DISCORD: discord_responder,
            ChannelType.WHATSAPP: whatsapp_responder,
            ChannelType.HTTP: http_responder,
            ChannelType.WEBSOCKET: websocket_responder,
        }

        self._initialized = True

    def register_responder(self, channel: ChannelType, responder: Callable) -> Any:
        """Register a custom responder for a channel."""
        self._responders[channel] = responder
        logger.info(f"Registered custom responder for {channel.value}")

    def get_responder(self, channel: ChannelType) -> Optional[Callable]:
        """Get responder for a channel."""
        self._initialize_responders()
        return self._responders.get(channel)

    async def send(self, response: ResponseEvent) -> Any:
        """
        Send a response through the appropriate channel.

        Args:
            response: The response event to send
        """
        self._initialize_responders()

        responder = self._responders.get(response.channel)
        if responder:
            try:
                await responder(response)
                logger.debug(f"Sent response to {response.channel.value}:{response.channel_id}")
            except Exception as e:
                logger.error(f"Failed to send response via {response.channel.value}: {e}")
                raise
        else:
            logger.warning(f"No responder registered for {response.channel.value}")

    def format_for_channel(self, content: str, channel: ChannelType) -> str:
        """
        Format content for specific channel requirements.

        Different channels have different formatting needs:
        - Telegram: Markdown (limited)
        - Slack: Slack mrkdwn
        - Discord: Discord Markdown
        - WhatsApp: Plain text with basic formatting
        """
        if channel == ChannelType.TELEGRAM:
            # Telegram supports limited Markdown
            # Escape special characters that aren't part of formatting
            return content

        elif channel == ChannelType.SLACK:
            # Convert standard Markdown to Slack mrkdwn
            content = content.replace("**", "*")  # Bold
            content = content.replace("__", "_")  # Italic
            return content

        elif channel == ChannelType.DISCORD:
            # Discord supports full Markdown
            return content

        elif channel == ChannelType.WHATSAPP:
            # WhatsApp has limited formatting
            # *bold*, _italic_, ~strikethrough~, ```code```
            return content

        else:
            return content


# Singleton instance
_responder_registry: Optional[ChannelResponderRegistry] = None


def get_responder_registry() -> ChannelResponderRegistry:
    """Get the singleton responder registry."""
    global _responder_registry
    if _responder_registry is None:
        _responder_registry = ChannelResponderRegistry()
    return _responder_registry
