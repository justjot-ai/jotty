"""
WhatsApp App
============

Personal WhatsApp integration via QR code (like OpenClaw/Moltbot).
No business account needed.

Components:
- client.py - WhatsApp client (Baileys bridge)
- command.py - CLI command (/whatsapp)
- bridge.js - Node.js bridge to Baileys
"""

from .client import WhatsAppWebClient

__all__ = ["WhatsAppWebClient"]
