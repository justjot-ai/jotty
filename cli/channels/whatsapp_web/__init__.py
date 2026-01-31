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
"""

from .client import WhatsAppWebClient, WhatsAppWebMessage

__all__ = ["WhatsAppWebClient", "WhatsAppWebMessage"]
