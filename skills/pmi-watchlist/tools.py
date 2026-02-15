"""
PMI Watchlist Skill
===================

Create, manage, and monitor watchlists via PlanMyInvesting API.
"""

import logging
from typing import Any, Dict

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-watchlist")


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
async def list_watchlists_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all watchlists with their symbols.

    Args:
        params: Dictionary (no required params)

    Returns:
        Dictionary with:
            - watchlists (list): Watchlist objects with id, name, symbols
            - count (int): Number of watchlists
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading watchlists...")

    result = client.get("/v2/watchlists")
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to list watchlists"))

    watchlists = result.get("watchlists", result.get("data", []))
    return tool_response(watchlists=watchlists, count=len(watchlists))


@async_tool_wrapper(required_params=["name"])
async def create_watchlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new watchlist.

    Args:
        params: Dictionary containing:
            - name (str, required): Watchlist name
            - symbols (list, optional): Initial symbols to add

    Returns:
        Dictionary with:
            - watchlist_id (str): ID of created watchlist
            - name (str): Watchlist name
            - symbols (list): Symbols added to watchlist
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    name = params["name"]
    symbols = params.get("symbols", [])
    status.emit("Creating", f"Creating watchlist: {name}...")

    result = client.post(
        "/v2/watchlists",
        data={
            "name": name,
            "symbols": symbols,
        },
    )
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to create watchlist '{name}'"))

    return tool_response(
        watchlist_id=result.get("id", result.get("watchlist_id")),
        name=name,
        symbols=symbols,
    )


@async_tool_wrapper(required_params=["watchlist_id", "symbol"])
async def add_to_watchlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a symbol to an existing watchlist.

    Args:
        params: Dictionary containing:
            - watchlist_id (str, required): Watchlist ID
            - symbol (str, required): Symbol to add

    Returns:
        Dictionary with:
            - watchlist_id (str): Watchlist ID
            - symbol (str): Symbol that was added
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    watchlist_id = params["watchlist_id"]
    symbol = params["symbol"]
    status.emit("Adding", f"Adding {symbol} to watchlist...")

    result = client.post(
        f"/v2/watchlists/{watchlist_id}/symbols",
        data={
            "symbol": symbol,
        },
    )
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to add {symbol} to watchlist"))

    return tool_response(watchlist_id=watchlist_id, symbol=symbol)


@async_tool_wrapper(required_params=["watchlist_id", "symbol"])
async def remove_from_watchlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove a symbol from a watchlist.

    Args:
        params: Dictionary containing:
            - watchlist_id (str, required): Watchlist ID
            - symbol (str, required): Symbol to remove

    Returns:
        Dictionary with:
            - watchlist_id (str): Watchlist ID
            - symbol (str): Symbol that was removed
            - removed (bool): Whether removal succeeded
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    watchlist_id = params["watchlist_id"]
    symbol = params["symbol"]
    status.emit("Removing", f"Removing {symbol} from watchlist...")

    result = client.delete(f"/v2/watchlists/{watchlist_id}/symbols/{symbol}")
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to remove {symbol} from watchlist"))

    return tool_response(watchlist_id=watchlist_id, symbol=symbol, removed=True)


@async_tool_wrapper(required_params=["watchlist_id"])
async def refresh_watchlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refresh live prices for all symbols in a watchlist.

    Args:
        params: Dictionary containing:
            - watchlist_id (str, required): Watchlist ID to refresh

    Returns:
        Dictionary with:
            - watchlist_id (str): Watchlist ID
            - symbols (list): Symbol objects with refreshed ltp, change, change_percent
            - count (int): Number of symbols
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    watchlist_id = params["watchlist_id"]
    status.emit("Refreshing", "Refreshing watchlist prices...")

    result = client.get(f"/v2/watchlists/{watchlist_id}/refresh")
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to refresh watchlist"))

    return tool_response(
        watchlist_id=watchlist_id,
        symbols=result.get("symbols", result.get("data", [])),
        count=len(result.get("symbols", result.get("data", []))),
    )


__all__ = [
    "list_watchlists_tool",
    "create_watchlist_tool",
    "add_to_watchlist_tool",
    "remove_from_watchlist_tool",
    "refresh_watchlist_tool",
]
