"""Notification Aggregator Skill - route notifications to channels."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("notification-aggregator")
logger = logging.getLogger("jotty.skills.notification-aggregator")

LEVELS = {"info": "INFO", "warning": "WARNING", "error": "ERROR", "critical": "CRITICAL"}
LEVEL_ICONS = {"info": "[i]", "warning": "[!]", "error": "[X]", "critical": "[!!!]"}


def _format_notification(title: str, message: str, level: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "title": title,
        "message": message,
    }


def _send_console(notification: dict) -> bool:
    icon = LEVEL_ICONS.get(notification["level"].lower(), "[i]")
    ts = notification["timestamp"][:19]
    print(f"{icon} [{ts}] {notification['title']}: {notification['message']}")
    return True


def _send_log(notification: dict) -> bool:
    level_map = {"INFO": logging.INFO, "WARNING": logging.WARNING,
                 "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    log_level = level_map.get(notification["level"], logging.INFO)
    logger.log(log_level, "%s: %s", notification["title"], notification["message"])
    return True


def _send_webhook(notification: dict, url: str) -> bool:
    try:
        import requests
        resp = requests.post(url, json=notification, timeout=10)
        return resp.status_code < 400
    except Exception:
        return False


def _send_file(notification: dict, file_path: str) -> bool:
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(notification) + "\n")
        return True
    except Exception:
        return False


@tool_wrapper(required_params=["message"])
def send_notification_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Route a notification to specified channels."""
    status.set_callback(params.pop("_status_callback", None))
    message = params["message"]
    channels = params.get("channels", ["console"])
    level = params.get("level", "info").lower()
    title = params.get("title", "Notification")
    webhook_url = params.get("webhook_url", "")
    file_path = params.get("file_path", "")

    if level not in LEVELS:
        return tool_error(f"Invalid level: {level}. Use: {list(LEVELS.keys())}")

    notification = _format_notification(title, message, level)
    delivered = []
    failed = []

    for channel in channels:
        ch = channel.lower().strip()
        ok = False
        if ch == "console":
            ok = _send_console(notification)
        elif ch == "log":
            ok = _send_log(notification)
        elif ch == "webhook":
            if not webhook_url:
                failed.append({"channel": ch, "error": "webhook_url required"})
                continue
            ok = _send_webhook(notification, webhook_url)
        elif ch == "file":
            if not file_path:
                failed.append({"channel": ch, "error": "file_path required"})
                continue
            ok = _send_file(notification, file_path)
        else:
            failed.append({"channel": ch, "error": f"Unknown channel: {ch}"})
            continue

        if ok:
            delivered.append(ch)
        else:
            failed.append({"channel": ch, "error": "delivery failed"})

    return tool_response(
        delivered=delivered, failed=failed,
        notification=notification,
        total_channels=len(channels),
    )


__all__ = ["send_notification_tool"]
