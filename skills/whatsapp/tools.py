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

import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from Jotty.core.infrastructure.utils.api_client import BaseAPIClient
from Jotty.core.infrastructure.utils.async_utils import run_sync

# Use centralized utilities
from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Load environment variables

# Status emitter for progress updates
status = SkillStatus("whatsapp")

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
    client = WhatsAppAPIClient(params.get("phone_id"), params.get("token"))
    if not client.phone_id or not client.token:
        return None, tool_error(
            "WhatsApp credentials required. Set WHATSAPP_PHONE_ID and WHATSAPP_TOKEN env vars"
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
                timeout=60,
            )

        if response.status_code in [200, 201]:
            result = response.json()
            return tool_response(media_id=result.get("id"))
        else:
            return tool_error(response.text)
    except Exception as e:
        return tool_error(str(e))


@async_tool_wrapper(required_params=["to", "message"])
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
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": params.get("preview_url", False), "body": params["message"]},
    }

    logger.info(f"Sending WhatsApp message to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)

    return result


@async_tool_wrapper(required_params=["to"])
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
    status.set_callback(params.pop("_status_callback", None))

    if not params.get("image_url") and not params.get("image_path"):
        return tool_error("image_url or image_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

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

    payload = {"messaging_product": "whatsapp", "to": to, "type": "image", "image": image_payload}

    logger.info(f"Sending WhatsApp image to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)

    return result


@async_tool_wrapper(required_params=["to"])
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
    status.set_callback(params.pop("_status_callback", None))

    if not params.get("document_url") and not params.get("document_path"):
        return tool_error("document_url or document_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])
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
        "document": doc_payload,
    }

    logger.info(f"Sending WhatsApp document to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)

    return result


@async_tool_wrapper(required_params=["to", "template_name"])
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
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])
    template_name = params["template_name"]

    template_payload = {
        "name": template_name,
        "language": {"code": params.get("language_code", "en_US")},
    }

    if params.get("components"):
        template_payload["components"] = params["components"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": template_payload,
    }

    logger.info(f"Sending WhatsApp template '{template_name}' to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None, to=to, template=template_name
        )

    return result


@async_tool_wrapper(required_params=["to", "latitude", "longitude"])
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
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

    location_payload = {"latitude": str(params["latitude"]), "longitude": str(params["longitude"])}

    if params.get("name"):
        location_payload["name"] = params["name"]
    if params.get("address"):
        location_payload["address"] = params["address"]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "location",
        "location": location_payload,
    }

    logger.info(f"Sending WhatsApp location to {to}")
    result = client._make_request("messages", json_data=payload)

    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)

    return result


@async_tool_wrapper(required_params=["message_id"])
async def mark_message_read_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark a message as read.

    Args:
        params: Dictionary containing:
            - message_id (str, required): WhatsApp message ID to mark as read

    Returns:
        Dictionary with success
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": params["message_id"],
    }

    result = client._make_request("messages", json_data=payload)
    return tool_response() if result.get("success") else result


@async_tool_wrapper(required_params=["to", "media_path"])
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
    status.set_callback(params.pop("_status_callback", None))
    try:
        from .providers import get_provider

        provider_name = params.get("provider", "auto")
        provider = get_provider(provider_name)
        return await provider.send_media(
            params["to"],
            params["media_path"],
            params.get("media_type", "image"),
            params.get("caption", ""),
        )
    except ImportError:
        # Fallback to Business API if providers not available
        if params.get("media_type", "image") == "document":
            return await send_whatsapp_document_tool(
                {
                    "to": params["to"],
                    "document_path": params["media_path"],
                    "caption": params.get("caption", ""),
                }
            )
        else:
            return await send_whatsapp_image_tool(
                {
                    "to": params["to"],
                    "image_path": params["media_path"],
                    "caption": params.get("caption", ""),
                }
            )


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
    status.set_callback(params.pop("_status_callback", None))
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
        return tool_response(connected=True, provider="business", phone=client.phone_id)


# Convenience functions for direct import using centralized run_sync
def send_whatsapp_message(phone: str, message: str, provider: str = "auto") -> Dict[str, Any]:
    """Send WhatsApp message synchronously."""
    return run_sync(
        send_whatsapp_message_tool({"to": phone, "message": message, "provider": provider})
    )


def send_whatsapp_media(
    phone: str, media_path: str, caption: str = "", provider: str = "auto"
) -> Dict[str, Any]:
    """Send WhatsApp media synchronously."""
    return run_sync(
        send_whatsapp_media_tool(
            {"to": phone, "media_path": media_path, "caption": caption, "provider": provider}
        )
    )


def get_whatsapp_status(provider: str = "auto") -> Dict[str, Any]:
    """Get WhatsApp status synchronously."""
    return run_sync(get_whatsapp_status_tool({"provider": provider}))


# =========================================================================
# Extended WhatsApp Tools (video, audio, reactions, contacts, interactive)
# =========================================================================


@async_tool_wrapper(required_params=["to"])
async def send_whatsapp_video_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a video via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - video_url (str, optional): URL of video
            - video_path (str, optional): Local path to video
            - caption (str, optional): Video caption

    Returns:
        Dictionary with success, message_id, to
    """
    status.set_callback(params.pop("_status_callback", None))

    if not params.get("video_url") and not params.get("video_path"):
        return tool_error("video_url or video_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

    if params.get("video_path") and not params.get("video_url"):
        upload_result = await _upload_media(client, params["video_path"])
        if not upload_result.get("success"):
            return upload_result
        video_payload = {"id": upload_result.get("media_id")}
    else:
        video_payload = {"link": params["video_url"]}

    if params.get("caption"):
        video_payload["caption"] = params["caption"]

    payload = {"messaging_product": "whatsapp", "to": to, "type": "video", "video": video_payload}

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)
    return result


@async_tool_wrapper(required_params=["to"])
async def send_whatsapp_audio_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an audio message via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - audio_url (str, optional): URL of audio
            - audio_path (str, optional): Local path to audio

    Returns:
        Dictionary with success, message_id, to
    """
    status.set_callback(params.pop("_status_callback", None))

    if not params.get("audio_url") and not params.get("audio_path"):
        return tool_error("audio_url or audio_path is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

    if params.get("audio_path") and not params.get("audio_url"):
        upload_result = await _upload_media(client, params["audio_path"])
        if not upload_result.get("success"):
            return upload_result
        audio_payload = {"id": upload_result.get("media_id")}
    else:
        audio_payload = {"link": params["audio_url"]}

    payload = {"messaging_product": "whatsapp", "to": to, "type": "audio", "audio": audio_payload}

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)
    return result


@async_tool_wrapper(required_params=["message_id", "emoji"])
async def send_whatsapp_reaction_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    React to a WhatsApp message with an emoji.

    Args:
        params: Dictionary containing:
            - message_id (str, required): ID of message to react to
            - emoji (str, required): Emoji to react with (e.g., thumbs up)

    Returns:
        Dictionary with success, message_id
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "type": "reaction",
        "reaction": {"message_id": params["message_id"], "emoji": params["emoji"]},
    }

    # Need a 'to' for the API but reaction goes to original sender
    to = params.get("to")
    if to:
        payload["to"] = _clean_phone(to)

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        return tool_response(reacted_to=params["message_id"], emoji=params["emoji"])
    return result


@async_tool_wrapper(required_params=["to", "message"])
async def send_whatsapp_reply_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reply to a specific WhatsApp message.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - message (str, required): Reply text
            - reply_to (str, required): Message ID to reply to

    Returns:
        Dictionary with success, message_id, to, replied_to
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])
    reply_to = params.get("reply_to")

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": params["message"]},
    }

    if reply_to:
        payload["context"] = {"message_id": reply_to}

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(
            message_id=messages[0].get("id") if messages else None, to=to, replied_to=reply_to
        )
    return result


@async_tool_wrapper()
async def get_whatsapp_profile_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get WhatsApp Business profile information.

    Args:
        params: Dictionary containing:
            - fields (str, optional): Comma-separated fields to retrieve

    Returns:
        Dictionary with success, profile data
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    fields = params.get("fields", "about,address,description,email,websites,profile_picture_url")

    try:
        url = f"{client.BASE_URL}/{client.phone_id}/whatsapp_business_profile"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {client.token}"},
            params={"fields": fields},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json().get("data", [{}])
            profile = data[0] if data else {}
            return tool_response(profile=profile)
        else:
            return tool_error(f"API error: {response.text}")

    except Exception as e:
        return tool_error(str(e))


@async_tool_wrapper(required_params=["to"])
async def send_whatsapp_contacts_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send contact cards via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - contacts (list, required): List of contact dicts with:
                - name: {first_name, last_name, formatted_name}
                - phones: [{phone, type}]
                - emails: [{email, type}] (optional)

    Returns:
        Dictionary with success, message_id, to
    """
    status.set_callback(params.pop("_status_callback", None))

    contacts = params.get("contacts")
    if not contacts:
        return tool_error("contacts list is required")

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])

    payload = {"messaging_product": "whatsapp", "to": to, "type": "contacts", "contacts": contacts}

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)
    return result


@async_tool_wrapper(required_params=["to"])
async def send_whatsapp_interactive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send interactive messages (buttons or lists) via WhatsApp.

    Args:
        params: Dictionary containing:
            - to (str, required): Recipient phone number
            - interactive_type (str, required): 'button' or 'list'
            - header (str, optional): Header text
            - body (str, required): Body text
            - footer (str, optional): Footer text
            - buttons (list, optional): For button type [{id, title}] (max 3)
            - sections (list, optional): For list type [{title, rows: [{id, title, description}]}]
            - button_text (str, optional): Button text for list type

    Returns:
        Dictionary with success, message_id, to
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    to = _clean_phone(params["to"])
    interactive_type = params.get("interactive_type", "button")

    interactive = {"type": interactive_type, "body": {"text": params.get("body", "")}}

    if params.get("header"):
        interactive["header"] = {"type": "text", "text": params["header"]}
    if params.get("footer"):
        interactive["footer"] = {"text": params["footer"]}

    if interactive_type == "button" and params.get("buttons"):
        interactive["action"] = {
            "buttons": [
                {"type": "reply", "reply": {"id": b.get("id", str(i)), "title": b["title"]}}
                for i, b in enumerate(params["buttons"][:3])
            ]
        }
    elif interactive_type == "list" and params.get("sections"):
        interactive["action"] = {
            "button": params.get("button_text", "Options"),
            "sections": params["sections"],
        }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }

    result = client._make_request("messages", json_data=payload)
    if result.get("success"):
        messages = result.get("messages", [])
        return tool_response(message_id=messages[0].get("id") if messages else None, to=to)
    return result


__all__ = [
    "send_whatsapp_message_tool",
    "send_whatsapp_image_tool",
    "send_whatsapp_document_tool",
    "send_whatsapp_template_tool",
    "send_whatsapp_location_tool",
    "send_whatsapp_media_tool",
    "get_whatsapp_status_tool",
    "mark_message_read_tool",
    # Extended tools
    "send_whatsapp_video_tool",
    "send_whatsapp_audio_tool",
    "send_whatsapp_reaction_tool",
    "send_whatsapp_reply_tool",
    "get_whatsapp_profile_tool",
    "send_whatsapp_contacts_tool",
    "send_whatsapp_interactive_tool",
    # Convenience functions
    "send_whatsapp_message",
    "send_whatsapp_media",
    "get_whatsapp_status",
]
