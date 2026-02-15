"""
PMI Alerts Skill
================

Create, manage, and monitor price/event alerts via PlanMyInvesting API.
"""

import logging
from typing import Dict, Any

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-alerts")


def _get_pmi_client():
    """Lazy import to avoid circular deps with hyphenated directory."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pmi_client",
        Path(__file__).resolve().parent.parent / "pmi-market-data" / "pmi_client.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlanMyInvestingClient()


def _require_client(client):
    """Return error dict if client is not configured."""
    return client.require_token()


@async_tool_wrapper()
async def list_alerts_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all active alerts.

    Args:
        params: Dictionary containing:
            - symbol (str, optional): Filter by symbol
            - alert_type (str, optional): Filter by type (price, volume, news)

    Returns:
        Dictionary with:
            - alerts (list): Alert objects with id, symbol, condition, value, status
            - count (int): Number of active alerts
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading alerts...")

    result = client.get("/api/alerts", params={
        "symbol": params.get("symbol", ""),
        "alert_type": params.get("alert_type", ""),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to list alerts"))

    alerts = result.get("alerts", result.get("data", []))
    return tool_response(alerts=alerts, count=len(alerts))


@async_tool_wrapper(required_params=["symbol", "condition", "value"])
async def create_alert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new price/event alert.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol
            - condition (str, required): Alert condition (above, below, crosses, percent_change)
            - value (float, required): Trigger value (price or percentage)
            - alert_type (str, optional): Type (price, volume) default "price"
            - message (str, optional): Custom alert message
            - notify_via (str, optional): Notification channel (telegram, email) default "telegram"

    Returns:
        Dictionary with:
            - alert_id (str): Unique alert identifier
            - symbol (str): Stock symbol
            - condition (str): Alert condition set
            - value (float): Trigger value
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    condition = params["condition"]
    value = params["value"]
    status.emit("Creating", f"Creating alert: {symbol} {condition} {value}...")

    result = client.post("/api/alerts", data={
        "symbol": symbol,
        "condition": condition,
        "value": value,
        "alert_type": params.get("alert_type", "price"),
        "message": params.get("message"),
        "notify_via": params.get("notify_via", "telegram"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to create alert for {symbol}"))

    return tool_response(
        alert_id=result.get("alert_id", result.get("id")),
        symbol=symbol,
        condition=condition,
        value=value,
    )


@async_tool_wrapper(required_params=["alert_id"])
async def delete_alert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete an alert.

    Args:
        params: Dictionary containing:
            - alert_id (str, required): Alert ID to delete

    Returns:
        Dictionary with:
            - alert_id (str): Deleted alert ID
            - deleted (bool): Whether deletion succeeded
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    alert_id = params["alert_id"]
    status.emit("Deleting", f"Deleting alert {alert_id}...")

    result = client.delete(f"/api/alerts/{alert_id}")
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to delete alert {alert_id}"))

    return tool_response(alert_id=alert_id, deleted=True)


@async_tool_wrapper()
async def get_alert_stats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get alert statistics (total, triggered, active).

    Args:
        params: Dictionary (no required params)

    Returns:
        Dictionary with:
            - total (int): Total number of alerts
            - active (int): Number of active alerts
            - triggered (int): Number of triggered alerts
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Getting alert statistics...")

    result = client.get("/api/alerts/stats")
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get alert stats"))

    return tool_response(
        total=result.get("total"),
        active=result.get("active"),
        triggered=result.get("triggered"),
        data=result.get("data"),
    )


__all__ = [
    "list_alerts_tool",
    "create_alert_tool",
    "delete_alert_tool",
    "get_alert_stats_tool",
]
