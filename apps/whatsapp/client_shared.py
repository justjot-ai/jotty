"""
WhatsApp Client (Migrated to Shared Components)
================================================

WhatsApp integration using shared UI components.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.shared import ChatInterface
from apps.shared.events import EventProcessor
from apps.shared.models import ChatSession, Message
from apps.shared.renderers import (  # WhatsApp uses similar format
    TelegramMessageRenderer,
    TelegramStatusRenderer,
)
from Jotty.sdk import Jotty, SDKEvent

logger = logging.getLogger(__name__)


class WhatsAppRenderer:
    """
    WhatsApp-specific renderer (extends Telegram renderer).

    WhatsApp and Telegram both use similar markdown formatting,
    so we reuse TelegramMessageRenderer with WhatsApp-specific tweaks.
    """

    def __init__(self, send_callback):
        """
        Initialize WhatsApp renderer.

        Args:
            send_callback: Function to send message via WhatsApp
        """
        # WhatsApp uses similar formatting to Telegram
        self.telegram_renderer = TelegramMessageRenderer(send_callback)

    def render_message(self, message: Message) -> None:
        """Render message for WhatsApp."""
        # WhatsApp has 65536 char limit (much higher than Telegram)
        # But we still use Telegram renderer for consistency
        self.telegram_renderer.render_message(message)

    def render_markdown(self, markdown: str) -> None:
        """Render markdown for WhatsApp."""
        self.telegram_renderer.render_markdown(markdown)

    # Delegate other methods to Telegram renderer
    def __getattr__(self, name):
        return getattr(self.telegram_renderer, name)


class WhatsAppChatInterface:
    """
    WhatsApp chat interface using shared components.

    Usage:
        wa = WhatsAppChatInterface(send_callback=send_wa_message)
        await wa.handle_message("Hello")
    """

    def __init__(self, send_callback, session_id: str = "default"):
        """
        Initialize WhatsApp interface.

        Args:
            send_callback: Function(text: str) to send message via WhatsApp
            session_id: Session ID for this conversation
        """
        self.send_callback = send_callback
        self.session_id = session_id

        # Create chat interface with WhatsApp renderer
        self.chat = ChatInterface(
            message_renderer=WhatsAppRenderer(send_callback),
            status_renderer=TelegramStatusRenderer(send_callback),
            input_handler=None,  # Not needed for bot
        )

        # Event processor
        self.event_processor = EventProcessor(self.chat)

        # SDK client
        self.sdk = Jotty()

        logger.info(f"WhatsApp interface initialized (session: {session_id})")

    async def handle_message(self, message_text: str, user_id: str, user_name: str) -> None:
        """
        Handle incoming WhatsApp message.

        Args:
            message_text: Message text from user
            user_id: WhatsApp user ID
            user_name: User's display name
        """
        # Add user message to chat
        user_msg = Message(
            role="user",
            content=message_text,
        )
        self.chat.add_message(user_msg)

        # Process via SDK with streaming
        try:
            async for event in self.sdk.chat_stream(
                message_text,
                session_id=self.session_id,
                user_id=user_id,
            ):
                # Process event (auto-sends to WhatsApp)
                await self.event_processor.process_event(event)

        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {e}", exc_info=True)
            from apps.shared.models import Error

            error = Error(
                message=str(e),
                error_type=type(e).__name__,
                recoverable=True,
            )
            self.chat.show_error(error)

    async def handle_command(self, command: str, user_id: str) -> None:
        """
        Handle WhatsApp command (e.g., /help, /status).

        Args:
            command: Command text (e.g., "/help")
            user_id: User ID
        """
        if command == "/help":
            help_text = """
*Jotty WhatsApp Bot*

Send any message to chat with Jotty AI.

*Commands:*
/help - Show this help
/status - Show bot status
/clear - Clear conversation history
/session - Show session info

*Examples:*
• Search for AI news
• Summarize this article: [url]
• Create a todo list for project X
"""
            self.send_callback(help_text)

        elif command == "/status":
            status_text = f"✅ Bot is active\nSession: {self.session_id}"
            self.send_callback(status_text)

        elif command == "/clear":
            self.chat.session.messages.clear()
            self.send_callback("🗑️ Conversation cleared")

        elif command == "/session":
            session = self.chat.session
            session_text = f"""
📊 *Session Info*

ID: {session.session_id}
Messages: {len(session.messages)}
Created: {session.created_at.strftime("%Y-%m-%d %H:%M")}
"""
            self.send_callback(session_text)

        else:
            self.send_callback(f"Unknown command: {command}")

    def get_session(self) -> ChatSession:
        """Get current chat session."""
        return self.chat.session


# Example integration with WhatsApp Web.js or similar
"""
from whatsapp_client_shared import WhatsAppChatInterface

# Initialize WhatsApp client
client = WhatsAppWebClient()
interfaces = {}  # chat_id -> WhatsAppChatInterface

async def handle_message(msg):
    chat_id = msg.from_

    # Get or create interface for this chat
    if chat_id not in interfaces:
        def send(text):
            client.send_message(chat_id, text)

        interfaces[chat_id] = WhatsAppChatInterface(
            send_callback=send,
            session_id=chat_id,
        )

    # Handle message
    if msg.body.startswith('/'):
        await interfaces[chat_id].handle_command(msg.body, chat_id)
    else:
        await interfaces[chat_id].handle_message(
            msg.body,
            user_id=chat_id,
            user_name=msg.author or "User",
        )

client.on('message', handle_message)
"""
