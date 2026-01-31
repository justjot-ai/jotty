"""
WhatsApp Skill
==============

Send and receive messages via WhatsApp Business Cloud API.

Setup:
1. Create Meta Business account at business.facebook.com
2. Create WhatsApp Business App at developers.facebook.com
3. Get Phone Number ID and Access Token
4. Set environment variables:
   - WHATSAPP_PHONE_ID: Your WhatsApp Business Phone Number ID
   - WHATSAPP_TOKEN: Permanent access token
"""
import os
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import base64
import mimetypes

# Load environment variables
try:
    from dotenv import load_dotenv
    current_file = Path(__file__).resolve()
    jotty_root = current_file.parent.parent.parent
    env_file = jotty_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

WHATSAPP_API_VERSION = "v18.0"
WHATSAPP_API_BASE = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"


class WhatsAppClient:
    """Helper class for WhatsApp API interactions."""

    def __init__(self, phone_id: Optional[str] = None, token: Optional[str] = None):
        self.phone_id = phone_id or os.getenv("WHATSAPP_PHONE_ID")
        self.token = token or os.getenv("WHATSAPP_TOKEN") or os.getenv("WHATSAPP_ACCESS_TOKEN")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _make_request(self, endpoint: str, method: str = "POST",
                      json_data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{WHATSAPP_API_BASE}/{self.phone_id}/{endpoint}"

        try:
            if method == "POST":
                response = requests.post(url, headers=self._get_headers(),
                                         json=json_data, timeout=30)
            elif method == "GET":
                response = requests.get(url, headers=self._get_headers(),
                                        params=json_data, timeout=30)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}

            if response.status_code in [200, 201]:
                result = response.json()
                return {"success": True, **result}
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text)
                    return {"success": False, "error": error_msg, "status_code": response.status_code}
                except Exception:
                    return {"success": False, "error": response.text, "status_code": response.status_code}

        except requests.exceptions.RequestException as e:
            logger.error(f"WhatsApp API error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


async def send_whatsapp_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a text message via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number with country code (e.g., '14155238886')
            - message (str, required): Message text
            - preview_url (bool, optional): Enable URL preview (default: False)
            - phone_id (str, optional): WhatsApp Phone Number ID (defaults to env)
            - token (str, optional): Access token (defaults to env)

    Returns:
        Dictionary with:
            - success (bool): Whether message was sent
            - message_id (str): WhatsApp message ID
            - error (str, optional): Error message if failed
    """
    try:
        to = params.get("to")
        message = params.get("message")

        if not to:
            return {"success": False, "error": "to parameter (phone number) is required"}
        if not message:
            return {"success": False, "error": "message parameter is required"}

        # Clean phone number (remove +, spaces, dashes)
        to = "".join(c for c in to if c.isdigit())

        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {
                "success": False,
                "error": "WhatsApp credentials required. Set WHATSAPP_PHONE_ID and WHATSAPP_TOKEN env vars"
            }

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": params.get("preview_url", False),
                "body": message
            }
        }

        logger.info(f"Sending WhatsApp message to {to}")
        result = client._make_request("messages", json_data=payload)

        if result.get("success"):
            messages = result.get("messages", [])
            return {
                "success": True,
                "message_id": messages[0].get("id") if messages else None,
                "to": to
            }

        return result

    except Exception as e:
        logger.error(f"WhatsApp send error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def send_whatsapp_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an image via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - image_url (str, optional): URL of image to send
            - image_path (str, optional): Local path to image (will be uploaded)
            - caption (str, optional): Image caption

    Returns:
        Dictionary with:
            - success (bool): Whether image was sent
            - message_id (str): WhatsApp message ID
            - error (str, optional): Error message if failed
    """
    try:
        to = params.get("to")
        image_url = params.get("image_url")
        image_path = params.get("image_path")
        caption = params.get("caption", "")

        if not to:
            return {"success": False, "error": "to parameter is required"}
        if not image_url and not image_path:
            return {"success": False, "error": "image_url or image_path is required"}

        to = "".join(c for c in to if c.isdigit())
        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {"success": False, "error": "WhatsApp credentials required"}

        # If local path, upload first
        if image_path and not image_url:
            upload_result = await _upload_media(client, image_path)
            if not upload_result.get("success"):
                return upload_result
            media_id = upload_result.get("media_id")

            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "image",
                "image": {"id": media_id}
            }
        else:
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "image",
                "image": {"link": image_url}
            }

        if caption:
            payload["image"]["caption"] = caption

        logger.info(f"Sending WhatsApp image to {to}")
        result = client._make_request("messages", json_data=payload)

        if result.get("success"):
            messages = result.get("messages", [])
            return {
                "success": True,
                "message_id": messages[0].get("id") if messages else None,
                "to": to
            }

        return result

    except Exception as e:
        logger.error(f"WhatsApp image send error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def send_whatsapp_document_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a document via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - document_url (str, optional): URL of document
            - document_path (str, optional): Local path to document
            - filename (str, optional): Display filename
            - caption (str, optional): Document caption

    Returns:
        Dictionary with:
            - success (bool): Whether document was sent
            - message_id (str): WhatsApp message ID
    """
    try:
        to = params.get("to")
        doc_url = params.get("document_url")
        doc_path = params.get("document_path")
        filename = params.get("filename")
        caption = params.get("caption", "")

        if not to:
            return {"success": False, "error": "to parameter is required"}
        if not doc_url and not doc_path:
            return {"success": False, "error": "document_url or document_path is required"}

        to = "".join(c for c in to if c.isdigit())
        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {"success": False, "error": "WhatsApp credentials required"}

        # If local path, upload first
        if doc_path and not doc_url:
            upload_result = await _upload_media(client, doc_path)
            if not upload_result.get("success"):
                return upload_result
            media_id = upload_result.get("media_id")

            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "document",
                "document": {"id": media_id}
            }
            if not filename:
                filename = Path(doc_path).name
        else:
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "document",
                "document": {"link": doc_url}
            }

        if filename:
            payload["document"]["filename"] = filename
        if caption:
            payload["document"]["caption"] = caption

        logger.info(f"Sending WhatsApp document to {to}")
        result = client._make_request("messages", json_data=payload)

        if result.get("success"):
            messages = result.get("messages", [])
            return {
                "success": True,
                "message_id": messages[0].get("id") if messages else None,
                "to": to
            }

        return result

    except Exception as e:
        logger.error(f"WhatsApp document send error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def send_whatsapp_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a template message via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - template_name (str, required): Template name
            - language_code (str, optional): Language code (default: 'en_US')
            - components (list, optional): Template components for variables

    Returns:
        Dictionary with:
            - success (bool): Whether message was sent
            - message_id (str): WhatsApp message ID
    """
    try:
        to = params.get("to")
        template_name = params.get("template_name")
        language_code = params.get("language_code", "en_US")
        components = params.get("components", [])

        if not to:
            return {"success": False, "error": "to parameter is required"}
        if not template_name:
            return {"success": False, "error": "template_name is required"}

        to = "".join(c for c in to if c.isdigit())
        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {"success": False, "error": "WhatsApp credentials required"}

        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code}
            }
        }

        if components:
            payload["template"]["components"] = components

        logger.info(f"Sending WhatsApp template '{template_name}' to {to}")
        result = client._make_request("messages", json_data=payload)

        if result.get("success"):
            messages = result.get("messages", [])
            return {
                "success": True,
                "message_id": messages[0].get("id") if messages else None,
                "to": to,
                "template": template_name
            }

        return result

    except Exception as e:
        logger.error(f"WhatsApp template send error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def send_whatsapp_location_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a location via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - latitude (float, required): Latitude
            - longitude (float, required): Longitude
            - name (str, optional): Location name
            - address (str, optional): Location address

    Returns:
        Dictionary with:
            - success (bool): Whether location was sent
            - message_id (str): WhatsApp message ID
    """
    try:
        to = params.get("to")
        latitude = params.get("latitude")
        longitude = params.get("longitude")
        name = params.get("name", "")
        address = params.get("address", "")

        if not to:
            return {"success": False, "error": "to parameter is required"}
        if latitude is None or longitude is None:
            return {"success": False, "error": "latitude and longitude are required"}

        to = "".join(c for c in to if c.isdigit())
        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {"success": False, "error": "WhatsApp credentials required"}

        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "location",
            "location": {
                "latitude": str(latitude),
                "longitude": str(longitude)
            }
        }

        if name:
            payload["location"]["name"] = name
        if address:
            payload["location"]["address"] = address

        logger.info(f"Sending WhatsApp location to {to}")
        result = client._make_request("messages", json_data=payload)

        if result.get("success"):
            messages = result.get("messages", [])
            return {
                "success": True,
                "message_id": messages[0].get("id") if messages else None,
                "to": to
            }

        return result

    except Exception as e:
        logger.error(f"WhatsApp location send error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _upload_media(client: WhatsAppClient, file_path: str) -> Dict[str, Any]:
    """Upload media to WhatsApp servers."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        url = f"{WHATSAPP_API_BASE}/{client.phone_id}/media"

        with open(path, "rb") as f:
            files = {"file": (path.name, f, mime_type)}
            data = {"messaging_product": "whatsapp"}

            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {client.token}"},
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code in [200, 201]:
            result = response.json()
            return {"success": True, "media_id": result.get("id")}
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def mark_message_read_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark a message as read.

    Args:
        params: Dictionary containing:
            - message_id (str, required): WhatsApp message ID to mark as read

    Returns:
        Dictionary with:
            - success (bool): Whether marked successfully
    """
    try:
        message_id = params.get("message_id")
        if not message_id:
            return {"success": False, "error": "message_id is required"}

        client = WhatsAppClient(params.get("phone_id"), params.get("token"))

        if not client.phone_id or not client.token:
            return {"success": False, "error": "WhatsApp credentials required"}

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }

        result = client._make_request("messages", json_data=payload)
        return {"success": result.get("success", False)}

    except Exception as e:
        return {"success": False, "error": str(e)}


__all__ = [
    "send_whatsapp_message_tool",
    "send_whatsapp_image_tool",
    "send_whatsapp_document_tool",
    "send_whatsapp_template_tool",
    "send_whatsapp_location_tool",
    "mark_message_read_tool"
]
