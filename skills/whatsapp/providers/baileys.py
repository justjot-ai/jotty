"""
Baileys WhatsApp Provider
=========================

Open-source WhatsApp provider using Baileys (like OpenClaw).
Baileys is a free, open-source WhatsApp Web API library.

This provider connects to a Baileys API server or uses the Node.js library directly.

Setup:
1. Install Baileys API server: https://github.com/WhiskeySockets/Baileys
2. Run the server: node server.js
3. Set BAILEYS_HOST=localhost and BAILEYS_PORT=3000
   Or set BAILEYS_SESSION_PATH to the session directory
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaileysProvider:
    """
    Baileys WhatsApp provider.

    Baileys is open-source and free (unlike Business API).
    Used by OpenClaw for their WhatsApp integration.
    """

    name = "baileys"

    def __init__(self):
        from . import get_config

        self.config = get_config()
        self.host = self.config.baileys_host or "localhost"
        self.port = self.config.baileys_port

    async def send_message(self, phone: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Send a text message via Baileys.

        Args:
            phone: Recipient phone number (with country code)
            message: Message text

        Returns:
            Dict with success, message_id, and optional error
        """
        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx package not installed. Install with: pip install httpx",
            }

        # Clean phone number
        phone = "".join(c for c in phone if c.isdigit())

        # Baileys expects JID format: phone@s.whatsapp.net
        jid = f"{phone}@s.whatsapp.net"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{self.host}:{self.port}/send/text",
                    json={"jid": jid, "message": message},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "message_id": result.get("key", {}).get("id"),
                        "to": phone,
                        "provider": self.name,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Baileys API error: {response.status_code} - {response.text}",
                    }

        except Exception as e:
            logger.error(f"Baileys send error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def send_media(
        self, phone: str, media_path: str, media_type: str = "image", caption: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Send media via Baileys.

        Args:
            phone: Recipient phone number
            media_path: Path to media file
            media_type: Type of media (image, video, audio, document)
            caption: Optional caption

        Returns:
            Dict with success, message_id, and optional error
        """
        try:
            import base64

            import httpx
        except ImportError:
            return {"success": False, "error": "httpx package not installed"}

        phone = "".join(c for c in phone if c.isdigit())
        jid = f"{phone}@s.whatsapp.net"

        media_file = Path(media_path)
        if not media_file.exists():
            return {"success": False, "error": f"Media file not found: {media_path}"}

        try:
            # Read and encode media
            media_data = base64.b64encode(media_file.read_bytes()).decode("utf-8")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{self.host}:{self.port}/send/media",
                    json={
                        "jid": jid,
                        "type": media_type,
                        "data": media_data,
                        "filename": media_file.name,
                        "caption": caption,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "message_id": result.get("key", {}).get("id"),
                        "to": phone,
                        "provider": self.name,
                    }
                else:
                    return {"success": False, "error": f"Baileys API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Baileys media send error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get Baileys connection status.

        Returns:
            Dict with success, connected status, and optional error
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx package not installed"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/status", timeout=10.0)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "connected": result.get("connected", False),
                        "phone": result.get("phone"),
                        "name": result.get("name"),
                        "provider": self.name,
                    }
                else:
                    return {"success": False, "error": f"Baileys API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Baileys status error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "connected": False}

    async def get_qr_code(self) -> Dict[str, Any]:
        """
        Get QR code for Baileys pairing.

        Returns:
            Dict with success, qr_code (base64), and optional error
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx package not installed"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/qr", timeout=30.0)

                if response.status_code == 200:
                    # QR code might be returned as image or base64
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type:
                        import base64

                        qr_data = base64.b64encode(response.content).decode("utf-8")
                        return {
                            "success": True,
                            "qr_code": qr_data,
                            "format": "png",
                            "provider": self.name,
                        }
                    else:
                        result = response.json()
                        return {
                            "success": True,
                            "qr_code": result.get("qr"),
                            "format": "text",
                            "provider": self.name,
                        }
                else:
                    return {"success": False, "error": f"Baileys API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Baileys QR error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
