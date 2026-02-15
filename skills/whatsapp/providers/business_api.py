"""
WhatsApp Business API Provider
==============================

Official Meta WhatsApp Business Cloud API provider.
Wraps the existing WhatsApp skill tools with the provider interface.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BusinessAPIProvider:
    """
    WhatsApp Business Cloud API provider.

    Uses the official Meta API. Requires:
    - WHATSAPP_PHONE_ID: Your WhatsApp Business Phone Number ID
    - WHATSAPP_TOKEN: Permanent access token
    """

    name = "business"

    def __init__(self):
        from . import get_config

        self.config = get_config()
        self.phone_id = self.config.business_phone_id
        self.token = self.config.business_token

    async def send_message(
        self, phone: str, message: str, preview_url: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Send a text message via WhatsApp Business API.

        Args:
            phone: Recipient phone number (with country code)
            message: Message text
            preview_url: Enable URL preview

        Returns:
            Dict with success, message_id, and optional error
        """
        # Import the existing tool
        from ..tools import send_whatsapp_message_tool

        result = await send_whatsapp_message_tool(
            {
                "to": phone,
                "message": message,
                "preview_url": preview_url,
                "phone_id": self.phone_id,
                "token": self.token,
            }
        )

        # Add provider info
        if result.get("success"):
            result["provider"] = self.name

        return result

    async def send_media(
        self, phone: str, media_path: str, media_type: str = "image", caption: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Send media via WhatsApp Business API.

        Args:
            phone: Recipient phone number
            media_path: Path to media file or URL
            media_type: Type of media (image, video, audio, document)
            caption: Optional caption

        Returns:
            Dict with success, message_id, and optional error
        """
        from ..tools import send_whatsapp_document_tool, send_whatsapp_image_tool

        # Determine if it's a URL or local path
        is_url = media_path.startswith("http://") or media_path.startswith("https://")

        if media_type in ("image", "video"):
            result = await send_whatsapp_image_tool(
                {
                    "to": phone,
                    "image_url" if is_url else "image_path": media_path,
                    "caption": caption,
                    "phone_id": self.phone_id,
                    "token": self.token,
                }
            )
        else:
            result = await send_whatsapp_document_tool(
                {
                    "to": phone,
                    "document_url" if is_url else "document_path": media_path,
                    "caption": caption,
                    "phone_id": self.phone_id,
                    "token": self.token,
                }
            )

        if result.get("success"):
            result["provider"] = self.name

        return result

    async def get_status(self) -> Dict[str, Any]:
        """
        Get WhatsApp Business API status.

        Returns:
            Dict with success, connected status, and optional error
        """
        import httpx

        if not self.phone_id or not self.token:
            return {
                "success": False,
                "error": "WhatsApp credentials not configured",
                "connected": False,
            }

        try:
            # Check if we can reach the API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://graph.facebook.com/v18.0/{self.phone_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "connected": True,
                        "phone": data.get("display_phone_number"),
                        "name": data.get("verified_name"),
                        "quality_rating": data.get("quality_rating"),
                        "provider": self.name,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "connected": False,
                    }

        except Exception as e:
            logger.error(f"Business API status error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "connected": False}
