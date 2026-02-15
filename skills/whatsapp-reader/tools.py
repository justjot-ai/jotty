"""
WhatsApp Reader Skill
=====================

Read messages from WhatsApp Web chats (Baileys session).
Use with Jotty swarm to read #channel/group messages and summarize learnings.
Requires WhatsApp Web session initialized in same process (/whatsapp login).
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper, async_tool_wrapper

logger = logging.getLogger(__name__)


def _get_client():
    """Get global WhatsApp Web client (set by CLI after /whatsapp login)."""
    try:
        from Jotty.cli.channels.whatsapp_web import get_global_whatsapp_client
        return get_global_whatsapp_client()
    except Exception as e:
        logger.debug(f"Could not get WhatsApp client: {e}")
        return None


@async_tool_wrapper()
async def read_whatsapp_chat_messages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read messages from a WhatsApp chat or group.

    Uses the global WhatsApp Web client (same session as /whatsapp login).
    Match chat by name (e.g. "my-ai-ml" or "#my-ai-ml") or by chat_id (JID).

    Args:
        params: dict with optional chat_name (str), chat_id (str), limit (int, default 100, max 500)

    Returns:
        dict with success, messages (list of {body, timestamp, fromMe, id}), chat_id, error
    """
    client = _get_client()
    if not client:
        return {
            "success": False,
            "messages": [],
            "error": "WhatsApp not connected. Run /whatsapp login first in this session, then run your task.",
        }
    if not getattr(client, "connected", False):
        return {
            "success": False,
            "messages": [],
            "error": "WhatsApp client not connected. Run /whatsapp login first.",
        }

    chat_name = (params.get("chat_name") or "").strip().strip("#") or None
    chat_id = params.get("chat_id") or None
    limit = min(int(params.get("limit") or 100), 500)

    if not chat_id and not chat_name:
        return {
            "success": False,
            "messages": [],
            "error": "Provide chat_name (e.g. 'my-ai-ml') or chat_id.",
        }

    try:
        result = await client.get_chat_messages(
            chat_id=chat_id,
            chat_name=chat_name,
            limit=limit,
        )
    except Exception as e:
        logger.exception("read_whatsapp_chat_messages failed")
        return {
            "success": False,
            "messages": [],
            "error": str(e),
        }

    error = result.get("error")
    messages = result.get("messages") or []
    return {
        "success": error is None,
        "messages": messages,
        "chat_id": result.get("chat_id"),
        "message_count": len(messages),
        "error": error,
    }


def _run_async(coro):
    """Run async coroutine from sync context if needed."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if asyncio.iscoroutine(coro):
        return loop.run_until_complete(coro)
    return coro


@tool_wrapper()
def read_whatsapp_chat_messages_tool_sync(params: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for read_whatsapp_chat_messages_tool."""
    return _run_async(read_whatsapp_chat_messages_tool(params))


@async_tool_wrapper()
async def summarize_whatsapp_chat_learnings_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read messages from a WhatsApp chat and summarize key learnings.

    Uses the summarize skill to produce a summary. Good for channels like #my-ai-ml.

    Args:
        params: chat_name (str, required), limit (int, default 200), length (str), style (str)

    Returns:
        dict with success, summary (str), message_count, error
    """
    chat_name = (params.get("chat_name") or "").strip().strip("#")
    if not chat_name:
        return {
            "success": False,
            "summary": "",
            "message_count": 0,
            "error": "chat_name is required (e.g. 'my-ai-ml').",
        }

    limit = min(int(params.get("limit") or 200), 500)
    length = params.get("length") or "medium"
    style = params.get("style") or "bullet"

    read_result = await read_whatsapp_chat_messages_tool({
        "chat_name": chat_name,
        "limit": limit,
    })

    if not read_result.get("success"):
        return {
            "success": False,
            "summary": "",
            "message_count": 0,
            "error": read_result.get("error", "Failed to read messages."),
        }

    messages = read_result.get("messages") or []
    if not messages:
        return {
            "success": True,
            "summary": "No messages found in this chat (or chat name did not match).",
            "message_count": 0,
        }

    # Build text: newest last, with sender and timestamp
    lines = []
    for m in messages:
        body = (m.get("body") or "").strip()
        if not body:
            continue
        ts = m.get("timestamp") or 0
        from_me = m.get("fromMe", False)
        prefix = "[Me]" if from_me else "[Other]"
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(ts) if ts else ""
            lines.append(f"{prefix} {dt}: {body}")
        except Exception:
            lines.append(f"{prefix}: {body}")

    combined = "\n\n".join(lines)
    if len(combined) > 120000:
        combined = combined[-120000:] + "\n\n[... truncated for length ...]"

    # Use summarize skill if available (via registry)
    try:
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        reg = get_skills_registry()
        reg.init()
        skill = reg.get_skill("summarize")
        if skill and skill.tools:
            summarize_text_tool = skill.tools.get("summarize_text_tool")
            if summarize_text_tool:
                summary_result = summarize_text_tool({
                    "text": combined,
                    "length": length,
                    "style": style,
                })
                if summary_result and summary_result.get("success"):
                    return {
                        "success": True,
                        "summary": summary_result.get("summary", ""),
                        "message_count": len(messages),
                    }
    except Exception as e:
        logger.warning(f"Summarize skill not available, returning raw excerpt: {e}")

    # Fallback: first 2000 chars as "summary" if summarize skill missing
    return {
        "success": True,
        "summary": combined[:2000] + ("..." if len(combined) > 2000 else ""),
        "message_count": len(messages),
        "note": "Summarize skill not used; excerpt only.",
    }


@tool_wrapper()
def summarize_whatsapp_chat_learnings_tool_sync(params: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for summarize_whatsapp_chat_learnings_tool."""
    return _run_async(summarize_whatsapp_chat_learnings_tool(params))
