"""
Messaging Tools - Multi-Channel Message Delivery
================================================

Comprehensive messaging and delivery utilities for multiple channels.

Modules:
- manager: OutputChannelManager - Multi-channel delivery (Telegram, WhatsApp, Email, Slack, etc.)

Individual skills in this package:
- telegram-sender: Telegram message delivery
- whatsapp: WhatsApp message delivery
- email skills: Email delivery
- slack skills: Slack message delivery

Usage:
    # Multi-channel delivery
    from skills.messaging_tools import OutputChannelManager, OutputChannel

    manager = OutputChannelManager()

    # Send to Telegram
    result = manager.send_to_telegram(
        file_path="report.pdf",
        caption="Check out this report!"
    )

    # Send to multiple channels
    results = manager.send_to_all(
        file_path="report.pdf",
        channels=["telegram", "whatsapp"],
        caption="New report available"
    )
"""

from .manager import (
    ChannelDeliveryResult,
    OutputChannel,
    OutputChannelManager,
)

__all__ = [
    "OutputChannel",
    "OutputChannelManager",
    "ChannelDeliveryResult",
]
