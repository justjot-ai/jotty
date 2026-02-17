"""
Messaging Tools Skill - Multi-channel message delivery.

Exposes OutputChannelManager as registry tools for Telegram, WhatsApp, and multi-channel send.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    result_to_tool_dict,
    tool_error,
)

status = SkillStatus("messaging-tools")

_manager: Optional[Any] = None


def _get_manager() -> Any:
    global _manager
    if _manager is None:
        from .manager import OutputChannelManager

        _manager = OutputChannelManager()
    return _manager


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert ChannelDeliveryResult to dict for tool response."""
    return result_to_tool_dict(result, include=("channel", "message_id"))


@async_tool_wrapper()
async def send_to_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a message or file to Telegram."""
    status.set_callback(params.pop("_status_callback", None))
    if not params.get("file_path") and not params.get("message"):
        return tool_error("Either file_path or message must be provided")
    status.sending("Telegram")
    manager = _get_manager()
    result = await manager.send_to_telegram(
        file_path=params.get("file_path"),
        message=params.get("message"),
        caption=params.get("caption"),
        chat_id=params.get("chat_id"),
        parse_mode=params.get("parse_mode", "HTML"),
    )
    return _result_to_dict(result)


@async_tool_wrapper(required_params=["to"])
async def send_to_whatsapp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a message or file to WhatsApp."""
    status.set_callback(params.pop("_status_callback", None))
    if not params.get("file_path") and not params.get("message"):
        return tool_error("Either file_path or message must be provided")
    status.sending("WhatsApp")
    manager = _get_manager()
    result = await manager.send_to_whatsapp(
        to=params["to"],
        file_path=params.get("file_path"),
        message=params.get("message"),
        caption=params.get("caption"),
        provider=params.get("provider", "auto"),
    )
    return _result_to_dict(result)


@async_tool_wrapper(required_params=["channels"])
async def send_to_all_channels_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a message or file to multiple channels (Telegram, WhatsApp, etc.)."""
    status.set_callback(params.pop("_status_callback", None))
    manager = _get_manager()
    channel_params = {}
    for key in ("telegram_chat_id", "telegram_parse_mode", "whatsapp_to", "whatsapp_provider"):
        if params.get(key) is not None:
            channel_params[key] = params[key]
    results = await manager.send_to_all(
        channels=params["channels"],
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
