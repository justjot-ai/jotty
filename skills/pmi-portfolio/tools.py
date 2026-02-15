"""
PMI Portfolio Skill
===================

Portfolio holdings, P&L summary, cash balance, and account limits
from PlanMyInvesting API.
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
status = SkillStatus("pmi-portfolio")


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
async def get_portfolio_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current portfolio holdings with live prices.

    Args:
        params: Dictionary containing:
            - broker (str, optional): Filter by broker name
            - include_closed (bool, optional): Include closed positions (default False)

    Returns:
        Dictionary with:
            - holdings (list): Portfolio holdings with symbol, quantity, avg_price, ltp, pnl
            - total_value (float): Total portfolio market value
            - total_pnl (float): Total profit/loss across all holdings
            - count (int): Number of holdings
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading portfolio holdings...")

    result = client.get(
        "/v2/portfolio",
        params={
            "broker": params.get("broker", ""),
            "include_closed": params.get("include_closed", False),
        },
    )
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get portfolio"))

    return tool_response(
        holdings=result.get("holdings", result.get("data", [])),
        total_value=result.get("total_value"),
        total_pnl=result.get("total_pnl"),
        count=len(result.get("holdings", result.get("data", []))),
    )


@async_tool_wrapper()
async def get_pnl_summary_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get profit & loss summary across all portfolios.

    Args:
        params: Dictionary containing:
            - period (str, optional): Period filter (today, week, month, year, all)

    Returns:
        Dictionary with:
            - realized_pnl (float): Realized profit/loss from closed positions
            - unrealized_pnl (float): Unrealized profit/loss from open positions
            - total_pnl (float): Total profit/loss (realized + unrealized)
            - day_pnl (float): Today's profit/loss
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Calculating", "Computing P&L summary...")

    result = client.post(
        "/v2/get_pnl_summary",
        data={
            "period": params.get("period", "all"),
        },
    )
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get P&L summary"))

    return tool_response(
        realized_pnl=result.get("realized_pnl"),
        unrealized_pnl=result.get("unrealized_pnl"),
        total_pnl=result.get("total_pnl"),
        day_pnl=result.get("day_pnl"),
        data=result.get("data"),
    )


__all__ = [
    "get_portfolio_tool",
    "get_pnl_summary_tool",
]
