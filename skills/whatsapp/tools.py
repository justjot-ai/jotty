"""
WhatsApp Skill
==============

Send and receive messages via WhatsApp Business Cloud API.
Refactored to use Jotty core utilities.

Setup:
1. Create Meta Business account at business.facebook.com
2. Create WhatsApp Business App at developers.facebook.com
3. Get Phone Number ID and Access Token
4. Set environment variables:
   - WHATSAPP_PHONE_ID: Your WhatsApp Business Phone Number ID
   - WHATSAPP_TOKEN: Permanent access token
"""

import os
import logging
import mimetypes
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Use centralized utilities
from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.api_client import BaseAPIClient
from Jotty.core.utils.tool_helpers import (
    tool_response, tool_error, async_tool_wrapper
)
from Jotty.core.utils.async_utils import run_sync

# Load environment variables
load_jotty_env()

logger = logging.getLogger(__name__)

WHATSAPP_API_VERSION = "v18.0"


class WhatsAppAPIClient(BaseAPIClient):
    """WhatsApp API client using base utilities."""

    BASE_URL = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
    AUTH_PREFIX = "Bearer"
    TOKEN_ENV_VAR = "WHATSAPP_TOKEN"
    TOKEN_CONFIG_PATH = ".config/whatsapp/token"

    def __init__(self, phone_id: Optional[str] = None, token: Optional[str] = None):
        self.phone_id = phone_id or os.getenv("WHATSAPP_PHONE_ID")
        super().__init__(token or os.getenv("WHATSAPP_TOKEN") or os.getenv("WHATSAPP_ACCESS_TOKEN"))

    def _build_url(self, endpoint: str) -> str:
        """Build full URL with phone_id."""
        if endpoint.startswith("http"):
            return endpoint
        base = self.BASE_URL.rstrip("/")
        endpoint = endpoint.lstrip("/")
        return f"{base}/{self.phone_id}/{endpoint}"


def _get_client(params: Dict[str, Any]) -> tuple:
    """Get WhatsApp client, returning (client, error) tuple."""
    client = WhatsAppAPIClient(params.get('phone_id'), params.get('token'))
    if not client.phone_id or not client.token:
        return None, tool_error(
            'WhatsApp credentials required. Set WHATSAPP_PHONE_ID and WHATSAPP_TOKEN env vars'
        )
    return client, None


def _clean_phone(phone: str) -> str:
    """Clean phone number, removing +, spaces, dashes."""
    return "".join(c for c in phone if c.isdigit())


async def _upload_media(client: WhatsAppAPIClient, file_path: str) -> Dict[str, Any]:
    """Upload media to WhatsApp servers."""
    path = Path(file_path)
    if not path.exists():
        return tool_error(f"File not found: {file_path}")

    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    url = f"{client.BASE_URL}/{client.phone_id}/media"

    try:
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
            return tool_response(media_id=result.get("id"))
        else:
            return tool_error(response.text)
    except Exception as e:
        return tool_error(str(e))


@async_tool_wrapper(required_params=['to', 'message'])
async def send_whatsapp_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a text message via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number with country code
            - message (str, required): Message text
            - preview_url (bool, optional): Enable URL preview (default: False)
            - phone_id (str, optional): WhatsApp Phone Number ID
            - token (str, optional): Access token

    Returns:
        Dictionary with success, message_id, to
    """
    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params['to'])

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": params.get("preview_url", False),
            "body": params['message']
        }
    }

    logger.info(f"Sending WhatsApp message to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None,
            to=to
        )

    return result


@async_tool_wrapper(required_params=['to'])
async def send_whatsapp_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an image via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - image_url (str, optional): URL of image to send
            - image_path (str, optional): Local path to image
            - caption (str, optional): Image caption

    Returns:
        Dictionary with success, message_id, to
    """
    if not params.get("image_url") and not params.get("image_path"):
        return tool_error("image_url or image_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params['to'])

    # If local path, upload first
    if params.get("image_path") and not params.get("image_url"):
        upload_result = await _upload_media(client, params["image_path"])
        if not upload_result.get("success"):
            return upload_result
        image_payload = {"id": upload_result.get("media_id")}
    else:
        image_payload = {"link": params["image_url"]}

    if params.get("caption"):
        image_payload["caption"] = params["caption"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": image_payload
    }

    logger.info(f"Sending WhatsApp image to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None,
            to=to
        )

    return result


@async_tool_wrapper(required_params=['to'])
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
        Dictionary with success, message_id, to
    """
    if not params.get("document_url") and not params.get("document_path"):
        return tool_error("document_url or document_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params['to'])
    filename = params.get("filename")

    # If local path, upload first
    if params.get("document_path") and not params.get("document_url"):
        upload_result = await _upload_media(client, params["document_path"])
        if not upload_result.get("success"):
            return upload_result
        doc_payload = {"id": upload_result.get("media_id")}
        if not filename:
            filename = Path(params["document_path"]).name
    else:
        doc_payload = {"link": params["document_url"]}

    if filename:
        doc_payload["filename"] = filename
    if params.get("caption"):
        doc_payload["caption"] = params["caption"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": doc_payload
    }

    logger.info(f"Sending WhatsApp document to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None,
            to=to
        )

    return result


@async_tool_wrapper(required_params=['to', 'template_name'])
async def send_whatsapp_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a template message via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - template_name (str, required): Template name
            - language_code (str, optional): Language code (default: 'en_US')
            - components (list, optional): Template components

    Returns:
        Dictionary with success, message_id, to, template
    """
    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params['to'])
    template_name = params['template_name']

    template_payload = {
        "name": template_name,
        "language": {"code": params.get("language_code", "en_US")}
    }

    if params.get("components"):
        template_payload["components"] = params["components"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": template_payload
    }

    logger.info(f"Sending WhatsApp template '{template_name}' to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None,
            to=to,
            template=template_name
        )

    return result


@async_tool_wrapper(required_params=['to', 'latitude', 'longitude'])
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
        Dictionary with success, message_id, to
    """
    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params['to'])

    location_payload = {
        "latitude": str(params['latitude']),
        "longitude": str(params['longitude'])
    }

    if params.get("name"):
        location_payload["name"] = params["name"]
    if params.get("address"):
        location_payload["address"] = params["address"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "location",
        "location": location_payload
    }

    logger.info(f"Sending WhatsApp location to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None,
            to=to
        )

    return result


@async_tool_wrapper(required_params=['message_id'])
async def mark_message_read_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark a message as read.

    Args:
        params: Dictionary containing:
            - message_id (str, required): WhatsApp message ID to mark as read

    Returns:
        Dictionary with success
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": params['message_id']
    }

    result = client._make_request("messages", json_data=payload)
    return tool_response() if result.get("success") else result


@async_tool_wrapper(required_params=['to', 'media_path'])
async def send_whatsapp_media_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send media via WhatsApp with automatic provider selection.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - media_path (str, required): Path to media file or URL
            - media_type (str, optional): "image", "video", "audio", "document"
            - caption (str, optional): Media caption
            - provider (str, optional): "baileys", "business", or "auto"

    Returns:
        Dictionary with success, message_id, provider
    """
    try:
        from .providers import get_provider

        provider_name = params.get("provider", "auto")
        provider = get_provider(provider_name)
        return await provider.send_media(
            params['to'],
            params['media_path'],
            params.get("media_type", "image"),
            params.get("caption", "")
        )
    except ImportError:
        # Fallback to Business API if providers not available
        if params.get("media_type", "image") == "document":
            return await send_whatsapp_document_tool({
                'to': params['to'],
                'document_path': params['media_path'],
                'caption': params.get('caption', '')
            })
        else:
            return await send_whatsapp_image_tool({
                'to': params['to'],
                'image_path': params['media_path'],
                'caption': params.get('caption', '')
            })


@async_tool_wrapper()
async def get_whatsapp_status_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get WhatsApp connection status.

    Args:
        params: Dictionary containing:
            - provider (str, optional): "baileys", "business", or "auto"

    Returns:
        Dictionary with success, connected, phone, name, provider
    """
    try:
        from .providers import get_provider

        provider_name = params.get("provider", "auto")
        provider = get_provider(provider_name)
        return await provider.get_status()
    except ImportError:
        # Check Business API status
        client, error = _get_client(params)
        if error:
            return tool_response(connected=False, provider="business", error="Not configured")
        return tool_response(
            connected=True,
            provider="business",
            phone=client.phone_id
        )


# Convenience functions for direct import using centralized run_sync
def send_whatsapp_message(phone: str, message: str, provider: str = "auto") -> Dict[str, Any]:
    """Send WhatsApp message synchronously."""
    return run_sync(send_whatsapp_message_tool({
        'to': phone,
        'message': message,
        'provider': provider
    }))


def send_whatsapp_media(phone: str, media_path: str, caption: str = "", provider: str = "auto") -> Dict[str, Any]:
    """Send WhatsApp media synchronously."""
    return run_sync(send_whatsapp_media_tool({
        'to': phone,
        'media_path': media_path,
        'caption': caption,
        'provider': provider
    }))


def get_whatsapp_status(provider: str = "auto") -> Dict[str, Any]:
    """Get WhatsApp status synchronously."""
    return run_sync(get_whatsapp_status_tool({
        'provider': provider
    }))


__all__ = [
    "send_whatsapp_message_tool",
    "send_whatsapp_image_tool",
    "send_whatsapp_document_tool",
    "send_whatsapp_template_tool",
    "send_whatsapp_location_tool",
    "send_whatsapp_media_tool",
    "get_whatsapp_status_tool",
    "mark_message_read_tool",
    # Convenience functions
    "send_whatsapp_message",
    "send_whatsapp_media",
    "get_whatsapp_status",
]
