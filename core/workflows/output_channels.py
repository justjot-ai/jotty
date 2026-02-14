#!/usr/bin/env python3
"""
Output Channels - Multi-Channel Delivery
=========================================

Delivers generated content to multiple channels (Telegram, WhatsApp, Email, etc.)
using existing Jotty skills.

Uses:
- telegram-sender skill for Telegram delivery
- whatsapp skill for WhatsApp delivery
- email skills for email delivery
- notion skills for Notion delivery
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OutputChannel(Enum):
    """Supported output channels."""
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    NOTION = "notion"


@dataclass
class ChannelDeliveryResult:
    """Result from channel delivery."""
    channel: str
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OutputChannelManager:
    """
    Manages multi-channel delivery of generated content.

    Uses existing Jotty skills:
    - telegram-sender: Telegram delivery
    - whatsapp: WhatsApp delivery
    - email: Email delivery
    - notion: Notion delivery

    Usage:
        manager = OutputChannelManager()

        # Send to Telegram
        result = manager.send_to_telegram(
            file_path="report.pdf",
            caption="Research Report"
        )

        # Send to multiple channels
        results = manager.send_to_all(
            file_path="report.pdf",
            channels=["telegram", "whatsapp"],
            caption="Check out this report!"
        )
    """

    def __init__(
        self,
        auto_load_skills: bool = True
    ):
        """
        Initialize output channel manager.

        Args:
            auto_load_skills: Auto-load Jotty skills registry
        """
        self.registry = None
        self.skills = {}

        if auto_load_skills:
            self._load_skills()

    def _load_skills(self):
        """Load required skills from Jotty registry."""
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            self.registry = get_skills_registry()
            self.registry.init()

            # Load telegram sender skill
            telegram_skill = self.registry.get_skill('telegram-sender')
            if telegram_skill and telegram_skill.tools:
                self.skills['telegram'] = telegram_skill.tools
                logger.info("✅ Loaded telegram-sender skill")

            # Load whatsapp skill
            whatsapp_skill = self.registry.get_skill('whatsapp')
            if whatsapp_skill and whatsapp_skill.tools:
                self.skills['whatsapp'] = whatsapp_skill.tools
                logger.info("✅ Loaded whatsapp skill")

            # Load notion skill
            notion_skill = self.registry.get_skill('notion')
            if notion_skill and notion_skill.tools:
                self.skills['notion'] = notion_skill.tools
                logger.info("✅ Loaded notion skill")

        except Exception as e:
            logger.warning(f"Could not load skills registry: {e}")
            logger.warning("Channel delivery will not be available")

    def send_to_telegram(
        self,
        file_path: Optional[str] = None,
        message: Optional[str] = None,
        caption: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML"
    ) -> ChannelDeliveryResult:
        """
        Send to Telegram.

        Args:
            file_path: Path to file to send (optional)
            message: Text message to send (optional)
            caption: Caption for file
            chat_id: Telegram chat ID (default: from env)
            parse_mode: HTML or Markdown

        Returns:
            ChannelDeliveryResult
        """
        if 'telegram' not in self.skills:
            return ChannelDeliveryResult(
                channel="telegram",
                success=False,
                error="telegram-sender skill not available"
            )

        try:
            # Send file if provided
            if file_path:
                send_tool = self.skills['telegram'].get('send_telegram_file_tool')
                if not send_tool:
                    return ChannelDeliveryResult(
                        channel="telegram",
                        success=False,
                        error="send_telegram_file_tool not found"
                    )

                import inspect
                params = {
                    'file_path': file_path,
                    'caption': caption or Path(file_path).stem,
                    'parse_mode': parse_mode
                }
                if chat_id:
                    params['chat_id'] = chat_id

                # Check if async
                if inspect.iscoroutinefunction(send_tool):
                    import asyncio
                    result = asyncio.run(send_tool(params))
                else:
                    result = send_tool(params)

            # Send message if provided
            elif message:
                send_tool = self.skills['telegram'].get('send_telegram_message_tool')
                if not send_tool:
                    return ChannelDeliveryResult(
                        channel="telegram",
                        success=False,
                        error="send_telegram_message_tool not found"
                    )

                import inspect
                params = {
                    'message': message,
                    'parse_mode': parse_mode
                }
                if chat_id:
                    params['chat_id'] = chat_id

                if inspect.iscoroutinefunction(send_tool):
                    import asyncio
                    result = asyncio.run(send_tool(params))
                else:
                    result = send_tool(params)

            else:
                return ChannelDeliveryResult(
                    channel="telegram",
                    success=False,
                    error="Either file_path or message must be provided"
                )

            if result.get('success'):
                logger.info(f"✅ Sent to Telegram")
                return ChannelDeliveryResult(
                    channel="telegram",
                    success=True,
                    message_id=result.get('message_id'),
                    metadata={'chat_id': result.get('chat_id')}
                )
            else:
                return ChannelDeliveryResult(
                    channel="telegram",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Telegram delivery failed: {e}")
            return ChannelDeliveryResult(
                channel="telegram",
                success=False,
                error=str(e)
            )

    def send_to_whatsapp(
        self,
        to: str,
        file_path: Optional[str] = None,
        message: Optional[str] = None,
        caption: Optional[str] = None,
        provider: str = "auto"
    ) -> ChannelDeliveryResult:
        """
        Send to WhatsApp.

        Args:
            to: Recipient phone number with country code (e.g., '14155238886')
            file_path: Path to file to send (optional)
            message: Text message to send (optional)
            caption: Caption for media
            provider: "baileys", "business", or "auto"

        Returns:
            ChannelDeliveryResult
        """
        if 'whatsapp' not in self.skills:
            return ChannelDeliveryResult(
                channel="whatsapp",
                success=False,
                error="whatsapp skill not available"
            )

        try:
            # Send media if file provided
            if file_path:
                send_tool = self.skills['whatsapp'].get('send_whatsapp_media_tool')
                if not send_tool:
                    return ChannelDeliveryResult(
                        channel="whatsapp",
                        success=False,
                        error="send_whatsapp_media_tool not found"
                    )

                # Determine media type
                ext = Path(file_path).suffix.lower()
                if ext == '.pdf':
                    media_type = 'document'
                elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    media_type = 'image'
                elif ext in ['.mp4', '.avi', '.mov']:
                    media_type = 'video'
                else:
                    media_type = 'document'

                result = send_tool({
                    'to': to,
                    'media_path': file_path,
                    'media_type': media_type,
                    'caption': caption or Path(file_path).stem,
                    'provider': provider
                })

            # Send message if provided
            elif message:
                send_tool = self.skills['whatsapp'].get('send_whatsapp_message_tool')
                if not send_tool:
                    return ChannelDeliveryResult(
                        channel="whatsapp",
                        success=False,
                        error="send_whatsapp_message_tool not found"
                    )

                result = send_tool({
                    'to': to,
                    'message': message,
                    'provider': provider
                })

            else:
                return ChannelDeliveryResult(
                    channel="whatsapp",
                    success=False,
                    error="Either file_path or message must be provided"
                )

            if result.get('success'):
                logger.info(f"✅ Sent to WhatsApp ({result.get('provider')})")
                return ChannelDeliveryResult(
                    channel="whatsapp",
                    success=True,
                    message_id=result.get('message_id'),
                    metadata={
                        'provider': result.get('provider'),
                        'to': to
                    }
                )
            else:
                return ChannelDeliveryResult(
                    channel="whatsapp",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"WhatsApp delivery failed: {e}")
            return ChannelDeliveryResult(
                channel="whatsapp",
                success=False,
                error=str(e)
            )

    def send_to_all(
        self,
        channels: List[str],
        file_path: Optional[str] = None,
        message: Optional[str] = None,
        caption: Optional[str] = None,
        **channel_params
    ) -> Dict[str, ChannelDeliveryResult]:
        """
        Send to multiple channels.

        Args:
            channels: List of channel names (telegram, whatsapp, etc.)
            file_path: Path to file to send
            message: Text message to send
            caption: Caption for files
            **channel_params: Channel-specific parameters
                - telegram_chat_id: Telegram chat ID
                - whatsapp_to: WhatsApp recipient
                - whatsapp_provider: WhatsApp provider

        Returns:
            Dict mapping channel name to ChannelDeliveryResult
        """
        results = {}

        for channel in channels:
            channel_lower = channel.lower()

            if channel_lower == "telegram":
                results["telegram"] = self.send_to_telegram(
                    file_path=file_path,
                    message=message,
                    caption=caption,
                    chat_id=channel_params.get('telegram_chat_id'),
                    parse_mode=channel_params.get('telegram_parse_mode', 'HTML')
                )

            elif channel_lower == "whatsapp":
                whatsapp_to = channel_params.get('whatsapp_to')
                if not whatsapp_to:
                    results["whatsapp"] = ChannelDeliveryResult(
                        channel="whatsapp",
                        success=False,
                        error="whatsapp_to parameter required"
                    )
                else:
                    results["whatsapp"] = self.send_to_whatsapp(
                        to=whatsapp_to,
                        file_path=file_path,
                        message=message,
                        caption=caption,
                        provider=channel_params.get('whatsapp_provider', 'auto')
                    )

            else:
                logger.warning(f"Unknown channel: {channel}")
                results[channel_lower] = ChannelDeliveryResult(
                    channel=channel_lower,
                    success=False,
                    error=f"Channel not supported: {channel}"
                )

        return results

    def get_summary(self, results: Dict[str, ChannelDeliveryResult]) -> Dict[str, Any]:
        """
        Get summary of delivery results.

        Args:
            results: Dict from send_to_all()

        Returns:
            Summary dict with success counts
        """
        successful = [ch for ch, res in results.items() if res.success]
        failed = [ch for ch, res in results.items() if not res.success]

        return {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'successful_channels': successful,
            'failed_channels': failed,
            'errors': {
                ch: res.error
                for ch, res in results.items()
                if not res.success and res.error
            }
        }
