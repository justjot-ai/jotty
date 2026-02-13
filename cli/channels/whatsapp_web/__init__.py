"""
WhatsApp Web Channel
====================

Connect to WhatsApp using your personal account via QR code pairing.
Like OpenClaw's Baileys integration.

Usage:
    /whatsapp login    # Show QR code to scan
    /whatsapp send "message" --to 14155238886
    /whatsapp chats    # List recent chats
    /whatsapp logout   # Disconnect

Global client (for skills/swarm):
    When CLI runs /whatsapp login, the client is set as the global client.
    Skills (e.g. whatsapp-reader) can call get_global_whatsapp_client() to
    read chats/messages when running under the same process.
"""

from typing import Optional

from .client import WhatsAppWebClient, WhatsAppWebMessage

_global_client: Optional[WhatsAppWebClient] = None


def set_global_whatsapp_client(client: Optional[WhatsAppWebClient]) -> None:
    """Set the global WhatsApp Web client (used by CLI after /whatsapp login)."""
    global _global_client
    _global_client = client


def get_global_whatsapp_client() -> Optional[WhatsAppWebClient]:
    """Get the global WhatsApp Web client for skills (read chats/messages)."""
    return _global_client

__all__ = [
    "WhatsAppWebClient",
    "WhatsAppWebMessage",
    "get_global_whatsapp_client",
    "set_global_whatsapp_client",
]
