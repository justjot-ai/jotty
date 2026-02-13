"""
PMI Broker Skill
================

Manage broker connections and authentication via PlanMyInvesting API.
"""

import logging
from typing import Dict, Any

from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-broker")


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
async def list_brokers_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all connected brokers and their status.

    Args:
        params: Dictionary (no required params)

    Returns:
        Dictionary with brokers list and count
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading broker connections...")

    result = client.get("/v2/brokers")
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to list brokers"))

    brokers = result.get("brokers", result.get("data", []))
    return tool_response(brokers=brokers, count=len(brokers))


@async_tool_wrapper(required_params=["broker"])
async def get_broker_status_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed status for a specific broker connection.

    Args:
        params: Dictionary containing:
            - broker (str, required): Broker name (e.g. "zerodha", "angel")

    Returns:
        Dictionary with broker status, token_valid, last_login
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    broker = params["broker"]
    status.emit("Checking", f"Checking {broker} status...")

    result = client.get(f"/v2/brokers/{broker}/status")
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to get status for {broker}"))

    return tool_response(
        broker=broker,
        connected=result.get("connected"),
        token_valid=result.get("token_valid"),
        last_login=result.get("last_login"),
        expires_at=result.get("expires_at"),
        data=result.get("data"),
    )


@async_tool_wrapper(required_params=["broker"])
async def refresh_tokens_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refresh authentication tokens for a broker.

    Args:
        params: Dictionary containing:
            - broker (str, required): Broker name to refresh

    Returns:
        Dictionary with broker, refreshed status, new expiry
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    broker = params["broker"]
    status.emit("Refreshing", f"Refreshing {broker} tokens...")

    result = client.post(f"/v2/brokers/{broker}/refresh")
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to refresh tokens for {broker}"))

    return tool_response(
        broker=broker,
        refreshed=True,
        expires_at=result.get("expires_at"),
        token_valid=result.get("token_valid", True),
    )


__all__ = [
    "list_brokers_tool",
    "get_broker_status_tool",
    "refresh_tokens_tool",
]
