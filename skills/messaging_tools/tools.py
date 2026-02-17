"""
Messaging Tools Skill - Multi-channel message delivery.

Exposes OutputChannelManager as registry tools for Telegram, WhatsApp, and multi-channel send.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

_manager: Optional[Any] = None


def _get_manager() -> Any:
    global _manager
    if _manager is None:
        from .manager import OutputChannelManager

        _manager = OutputChannelManager()
    return _manager


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert ChannelDeliveryResult to dict for tool response."""
    return {
        "success": result.success,
        "channel": getattr(result, "channel", ""),
        "message_id": getattr(result, "message_id", None),
        "error": getattr(result, "error", None),
        "metadata": getattr(result, "metadata", None) or {},
    }


async def send_to_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message or file to Telegram.

    Args:
        params: file_path (optional), message (optional), caption, chat_id, parse_mode (HTML/Markdown).
                At least one of file_path or message is required.

    Returns:
        success, channel, message_id, error, metadata
    """
    manager = _get_manager()
    if not params.get("file_path") and not params.get("message"):
        return {"success": False, "error": "Either file_path or message must be provided"}
    result = await manager.send_to_telegram(
        file_path=params.get("file_path"),
        message=params.get("message"),
        caption=params.get("caption"),
        chat_id=params.get("chat_id"),
        parse_mode=params.get("parse_mode", "HTML"),
    )
    return _result_to_dict(result)


async def send_to_whatsapp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message or file to WhatsApp.

    Args:
        params: to (required, phone with country code), file_path, message, caption, provider (auto/baileys/business).
                At least one of file_path or message is required.

    Returns:
        success, channel, message_id, error, metadata
    """
    manager = _get_manager()
    to = params.get("to")
    if not to:
        return {"success": False, "error": "to (recipient phone with country code) is required"}
    if not params.get("file_path") and not params.get("message"):
        return {"success": False, "error": "Either file_path or message must be provided"}
    result = await manager.send_to_whatsapp(
        to=to,
        file_path=params.get("file_path"),
        message=params.get("message"),
        caption=params.get("caption"),
        provider=params.get("provider", "auto"),
    )
    return _result_to_dict(result)


async def send_to_all_channels_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message or file to multiple channels (Telegram, WhatsApp, etc.).

    Args:
        params: channels (required, list e.g. ["telegram","whatsapp"]), file_path, message, caption,
                telegram_chat_id, whatsapp_to, whatsapp_provider.

    Returns:
        success (true if any succeeded), summary (total, successful, failed, successful_channels, errors), results per channel
    """
    manager = _get_manager()
    channels = params.get("channels") or []
    if not channels:
        return {
            "success": False,
            "error": 'channels (list) is required, e.g. ["telegram", "whatsapp"]',
        }
    channel_params = {}
    if params.get("telegram_chat_id") is not None:
        channel_params["telegram_chat_id"] = params["telegram_chat_id"]
    if params.get("telegram_parse_mode") is not None:
        channel_params["telegram_parse_mode"] = params["telegram_parse_mode"]
    if params.get("whatsapp_to") is not None:
        channel_params["whatsapp_to"] = params["whatsapp_to"]
    if params.get("whatsapp_provider") is not None:
        channel_params["whatsapp_provider"] = params["whatsapp_provider"]
    results = await manager.send_to_all(
        channels=channels,
        file_path=params.get("file_path"),
        message=params.get("message"),
        caption=params.get("caption"),
        **channel_params,
    )
    summary = manager.get_summary(results)
    results_dict = {ch: _result_to_dict(res) for ch, res in results.items()}
    return {
        "success": summary.get("successful", 0) > 0,
        "summary": summary,
        "results": results_dict,
    }
