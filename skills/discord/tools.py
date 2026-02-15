"""
Discord Skill

Interact with Discord using the Discord API via requests.
Refactored to use Jotty core utilities.
"""

import logging
import urllib.parse
from typing import Any, Dict, Optional

import requests

from Jotty.core.infrastructure.utils.api_client import BaseAPIClient

# Use centralized utilities
from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    require_params,
    tool_error,
    tool_response,
    tool_wrapper,
)

# Load environment variables
load_jotty_env()

# Status emitter for progress updates
status = SkillStatus("discord")

logger = logging.getLogger(__name__)


class DiscordAPIClient(BaseAPIClient):
    """Discord API client using base utilities."""

    BASE_URL = "https://discord.com/api/v10"
    AUTH_PREFIX = "Bot"
    TOKEN_ENV_VAR = "DISCORD_BOT_TOKEN"
    TOKEN_CONFIG_PATH = ".config/discord/token"

    # Channel type mapping
    CHANNEL_TYPES = {
        0: "text",
        2: "voice",
        4: "category",
        5: "announcement",
        10: "announcement_thread",
        11: "public_thread",
        12: "private_thread",
        13: "stage_voice",
        14: "directory",
        15: "forum",
        16: "media",
    }


def _get_client(params: Dict[str, Any]) -> tuple:
    """Get Discord client, returning (client, error) tuple."""
    client = DiscordAPIClient(params.get("token"))
    if not client.token:
        return None, tool_error(
            "Discord token required. Set DISCORD_BOT_TOKEN env var or provide token parameter"
        )
    return client, None


@tool_wrapper()
def send_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message to a Discord channel.

    Args:
        params: Dictionary containing:
            - channel_id (str, required): Channel ID to send message to
            - content (str, required): Message content (up to 2000 characters)
            - token (str, optional): Discord bot token
            - tts (bool, optional): Whether this is a TTS message
            - embed (dict, optional): Embed object to include
            - embeds (list, optional): Array of embed objects (max 10)
            - message_reference (dict, optional): Message reference for replies

    Returns:
        Dictionary with success, id, channel_id, content, author, timestamp
    """
    status.set_callback(params.pop("_status_callback", None))

    channel_id = params.get("channel_id")
    content = params.get("content")

    if not channel_id:
        return tool_error("channel_id parameter is required")

    if not content and not params.get("embed") and not params.get("embeds"):
        return tool_error("content, embed, or embeds parameter is required")

    client, error = _get_client(params)
    if error:
        return error

    payload = {}

    if content:
        if len(content) > 2000:
            return tool_error("Message content cannot exceed 2000 characters")
        payload["content"] = content

    if params.get("tts"):
        payload["tts"] = params["tts"]

    if params.get("embed"):
        payload["embeds"] = [params["embed"]]
    elif params.get("embeds"):
        payload["embeds"] = params["embeds"][:10]

    if params.get("message_reference"):
        payload["message_reference"] = params["message_reference"]

    logger.info(f"Sending Discord message to channel {channel_id}")
    result = client._make_request(f"/channels/{channel_id}/messages", json_data=payload)

    if result.get("success"):
        return tool_response(
            id=result.get("id"),
            channel_id=result.get("channel_id"),
            content=result.get("content"),
            author=result.get("author"),
            timestamp=result.get("timestamp"),
            embeds=result.get("embeds", []),
        )

    return result


@tool_wrapper(required_params=["guild_id"])
def list_channels_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List channels in a Discord guild (server).

    Args:
        params: Dictionary containing:
            - guild_id (str, required): Guild ID to list channels from
            - token (str, optional): Discord bot token

    Returns:
        Dictionary with success, channels list, channel_count
    """
    status.set_callback(params.pop("_status_callback", None))

    guild_id = params["guild_id"]

    client, error = _get_client(params)
    if error:
        return error

    logger.info(f"Listing Discord channels for guild {guild_id}")

    # Direct request to get list response
    url = f"{client.BASE_URL}/guilds/{guild_id}/channels"
    response = requests.get(url, headers=client._get_headers(), timeout=30)

    if response.status_code == 200:
        channels = response.json()
        return tool_response(
            channels=[
                {
                    "id": ch.get("id"),
                    "name": ch.get("name"),
                    "type": client.CHANNEL_TYPES.get(ch.get("type"), ch.get("type")),
                    "type_id": ch.get("type"),
                    "position": ch.get("position"),
                    "parent_id": ch.get("parent_id"),
                    "topic": ch.get("topic"),
                    "nsfw": ch.get("nsfw", False),
                }
                for ch in channels
                if isinstance(ch, dict)
            ],
            channel_count=len(channels),
        )

    return client._handle_response(response)


@tool_wrapper(required_params=["channel_id"])
def read_messages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read recent messages from a Discord channel.

    Args:
        params: Dictionary containing:
            - channel_id (str, required): Channel ID to read messages from
            - token (str, optional): Discord bot token
            - limit (int, optional): Number of messages (default: 50, max: 100)
            - before/after/around (str, optional): Message ID for pagination

    Returns:
        Dictionary with success, messages list, message_count
    """
    status.set_callback(params.pop("_status_callback", None))

    channel_id = params["channel_id"]

    client, error = _get_client(params)
    if error:
        return error

    query_params = {"limit": min(params.get("limit", 50), 100)}
    for key in ("before", "after", "around"):
        if params.get(key):
            query_params[key] = params[key]

    logger.info(f"Reading messages from Discord channel {channel_id}")

    url = f"{client.BASE_URL}/channels/{channel_id}/messages"
    response = requests.get(url, headers=client._get_headers(), params=query_params, timeout=30)

    if response.status_code == 200:
        messages = response.json()
        return tool_response(
            messages=[
                {
                    "id": msg.get("id"),
                    "content": msg.get("content"),
                    "author": {
                        "id": msg.get("author", {}).get("id"),
                        "username": msg.get("author", {}).get("username"),
                        "discriminator": msg.get("author", {}).get("discriminator"),
                        "global_name": msg.get("author", {}).get("global_name"),
                        "bot": msg.get("author", {}).get("bot", False),
                    },
                    "timestamp": msg.get("timestamp"),
                    "edited_timestamp": msg.get("edited_timestamp"),
                    "attachments": msg.get("attachments", []),
                    "embeds": msg.get("embeds", []),
                    "reactions": msg.get("reactions", []),
                    "referenced_message": msg.get("referenced_message"),
                    "thread": msg.get("thread"),
                }
                for msg in messages
            ],
            message_count=len(messages),
        )

    return client._handle_response(response)


@tool_wrapper(required_params=["channel_id", "message_id", "emoji"])
def add_reaction_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an emoji reaction to a message.

    Args:
        params: Dictionary containing:
            - channel_id (str, required): Channel ID
            - message_id (str, required): Message ID to react to
            - emoji (str, required): Emoji (unicode or custom format name:id)
            - token (str, optional): Discord bot token

    Returns:
        Dictionary with success, channel_id, message_id, emoji
    """
    status.set_callback(params.pop("_status_callback", None))

    channel_id = params["channel_id"]
    message_id = params["message_id"]
    emoji = params["emoji"]

    client, error = _get_client(params)
    if error:
        return error

    encoded_emoji = urllib.parse.quote(emoji)

    logger.info(f"Adding reaction {emoji} to message {message_id}")
    result = client._make_request(
        f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/@me", method="PUT"
    )

    if result.get("success"):
        return tool_response(channel_id=channel_id, message_id=message_id, emoji=emoji)

    return result


@tool_wrapper(required_params=["user_id"])
def get_user_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get details about a Discord user.

    Args:
        params: Dictionary containing:
            - user_id (str, required): User ID to get info for
            - token (str, optional): Discord bot token

    Returns:
        Dictionary with success, user object
    """
    status.set_callback(params.pop("_status_callback", None))

    user_id = params["user_id"]

    client, error = _get_client(params)
    if error:
        return error

    logger.info(f"Getting Discord user info for {user_id}")
    result = client._make_request(f"/users/{user_id}", method="GET")

    if result.get("success"):
        avatar = result.get("avatar")
        return tool_response(
            user={
                "id": result.get("id"),
                "username": result.get("username"),
                "discriminator": result.get("discriminator"),
                "global_name": result.get("global_name"),
                "avatar": avatar,
                "avatar_url": (
                    f"https://cdn.discordapp.com/avatars/{result.get('id')}/{avatar}.png"
                    if avatar
                    else None
                ),
                "bot": result.get("bot", False),
                "system": result.get("system", False),
                "banner": result.get("banner"),
                "accent_color": result.get("accent_color"),
                "public_flags": result.get("public_flags"),
            }
        )

    return result


@tool_wrapper(required_params=["channel_id", "name"])
def create_thread_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a thread in a Discord channel.

    Args:
        params: Dictionary containing:
            - channel_id (str, required): Channel ID
            - name (str, required): Thread name (1-100 characters)
            - token (str, optional): Discord bot token
            - message_id (str, optional): Message ID to start thread from
            - auto_archive_duration (int, optional): Minutes (60, 1440, 4320, 10080)
            - type (int, optional): Thread type (10, 11, 12)
            - invitable (bool, optional): Allow non-mods to add users

    Returns:
        Dictionary with success, id, name, parent_id, owner_id, type
    """
    status.set_callback(params.pop("_status_callback", None))

    channel_id = params["channel_id"]
    name = params["name"]
    message_id = params.get("message_id")

    if len(name) > 100:
        return tool_error("Thread name cannot exceed 100 characters")

    client, error = _get_client(params)
    if error:
        return error

    payload = {"name": name}

    if params.get("auto_archive_duration"):
        valid_durations = [60, 1440, 4320, 10080]
        duration = params["auto_archive_duration"]
        if duration not in valid_durations:
            return tool_error(f"auto_archive_duration must be one of {valid_durations}")
        payload["auto_archive_duration"] = duration

    if message_id:
        endpoint = f"/channels/{channel_id}/messages/{message_id}/threads"
    else:
        endpoint = f"/channels/{channel_id}/threads"
        if params.get("type"):
            payload["type"] = params["type"]
        if params.get("invitable") is not None:
            payload["invitable"] = params["invitable"]

    logger.info(f"Creating Discord thread '{name}' in channel {channel_id}")
    result = client._make_request(endpoint, json_data=payload)

    if result.get("success"):
        return tool_response(
            id=result.get("id"),
            name=result.get("name"),
            parent_id=result.get("parent_id"),
            owner_id=result.get("owner_id"),
            type=result.get("type"),
            thread_metadata=result.get("thread_metadata"),
        )

    return result


__all__ = [
    "send_message_tool",
    "list_channels_tool",
    "read_messages_tool",
    "add_reaction_tool",
    "get_user_info_tool",
    "create_thread_tool",
]
