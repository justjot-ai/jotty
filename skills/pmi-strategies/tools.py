"""
PMI Strategies Skill
====================

List, run, and monitor trading strategies via PlanMyInvesting API.
"""

import logging
from typing import Dict, Any

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-strategies")


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
async def list_strategies_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all available trading strategies.

    Args:
        params: Dictionary containing:
            - active_only (bool, optional): Only show active strategies

    Returns:
        Dictionary with:
            - strategies (list): Strategy objects with id, name, active, description
            - count (int): Number of strategies
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading strategies...")

    result = client.get("/v2/strategies", params={
        "active_only": params.get("active_only", False),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to list strategies"))

    strategies = result.get("strategies", result.get("data", []))
    return tool_response(strategies=strategies, count=len(strategies))


@async_tool_wrapper(required_params=["strategy_id"])
async def run_strategy_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a trading strategy.

    Args:
        params: Dictionary containing:
            - strategy_id (str, required): Strategy ID to run
            - dry_run (bool, optional): Simulate without placing orders (default True)
            - symbols (list, optional): Override universe of symbols

    Returns:
        Dictionary with:
            - execution_id (str): Unique execution run identifier
            - strategy_id (str): Strategy that was executed
            - signals (list): Generated trading signals
            - orders_placed (int): Number of orders placed
            - dry_run (bool): Whether this was a simulation
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    strategy_id = params["strategy_id"]
    dry_run = params.get("dry_run", True)
    status.emit("Running", f"Executing strategy {strategy_id} (dry_run={dry_run})...")

    result = client.post(f"/v2/strategies/{strategy_id}/run", data={
        "dry_run": dry_run,
        "symbols": params.get("symbols"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to run strategy {strategy_id}"))

    return tool_response(
        execution_id=result.get("execution_id"),
        strategy_id=strategy_id,
        signals=result.get("signals", []),
        orders_placed=result.get("orders_placed", 0),
        dry_run=dry_run,
    )


@async_tool_wrapper(required_params=["strategy_id"])
async def get_strategy_status_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current status and performance of a strategy.

    Args:
        params: Dictionary containing:
            - strategy_id (str, required): Strategy ID

    Returns:
        Dictionary with:
            - strategy_id (str): Strategy identifier
            - name (str): Strategy name
            - active (bool): Whether strategy is active
            - total_trades (int): Total number of trades
            - win_rate (float): Win rate percentage
            - pnl (float): Profit/loss from this strategy
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    strategy_id = params["strategy_id"]
    status.emit("Fetching", f"Getting status for strategy {strategy_id}...")

    result = client.get(f"/v2/strategies/status/{strategy_id}")
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to get strategy status"))

    return tool_response(
        strategy_id=strategy_id,
        name=result.get("name"),
        active=result.get("active"),
        total_trades=result.get("total_trades"),
        win_rate=result.get("win_rate"),
        pnl=result.get("pnl"),
        data=result.get("data"),
    )


@async_tool_wrapper()
async def generate_signals_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate trading signals from all active strategies.

    Args:
        params: Dictionary containing:
            - symbols (list, optional): Filter signals for specific symbols
            - strategy_id (str, optional): Generate signals for a specific strategy only

    Returns:
        Dictionary with:
            - signals (list): Signal objects with symbol, action, price, strategy
            - count (int): Number of signals generated
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Generating", "Generating trading signals...")

    result = client.post("/v2/strategies/generate-signals", data={
        "symbols": params.get("symbols"),
        "strategy_id": params.get("strategy_id"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to generate signals"))

    signals = result.get("signals", result.get("data", []))
    return tool_response(signals=signals, count=len(signals))


__all__ = [
    "list_strategies_tool",
    "run_strategy_tool",
    "get_strategy_status_tool",
    "generate_signals_tool",
]
