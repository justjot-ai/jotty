"""
PMI Trading Skill
=================

Place orders, manage positions, and track order status via PlanMyInvesting API.
"""

import logging
from typing import Dict, Any

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-trading")


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


@async_tool_wrapper(required_params=["symbol", "quantity", "order_type", "transaction_type"])
async def place_order_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Place a trading order.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol
            - quantity (int, required): Number of shares
            - order_type (str, required): MARKET, LIMIT, SL, SL-M
            - transaction_type (str, required): BUY or SELL
            - price (float, optional): Price for LIMIT/SL orders
            - trigger_price (float, optional): Trigger price for SL orders
            - broker (str, optional): Broker to use
            - product (str, optional): CNC, MIS, NRML (default CNC)

    Returns:
        Dictionary with:
            - order_id (str): Unique order identifier
            - status (str): Order status (placed, rejected, etc.)
            - symbol (str): Stock symbol
            - quantity (int): Number of shares ordered
            - transaction_type (str): BUY or SELL
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    txn_type = params["transaction_type"]
    qty = params["quantity"]
    status.emit("Placing", f"Placing {txn_type} order for {qty} {symbol}...")

    result = client.post("/v2/place_order", data={
        "symbol": symbol,
        "quantity": qty,
        "order_type": params["order_type"],
        "transaction_type": txn_type,
        "price": params.get("price"),
        "trigger_price": params.get("trigger_price"),
        "broker": params.get("broker"),
        "product": params.get("product", "CNC"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to place {txn_type} order for {symbol}"))

    return tool_response(
        order_id=result.get("order_id"),
        status=result.get("status", "placed"),
        symbol=symbol,
        quantity=qty,
        transaction_type=txn_type,
    )


@async_tool_wrapper(required_params=["symbol", "quantity"])
async def place_smart_order_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Place a smart order with automatic price optimization and bracket logic.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol
            - quantity (int, required): Number of shares
            - transaction_type (str, optional): BUY or SELL (default BUY)
            - target_percent (float, optional): Target profit % (default 2.0)
            - stoploss_percent (float, optional): Stop loss % (default 1.0)
            - broker (str, optional): Broker to use

    Returns:
        Dictionary with:
            - order_id (str): Unique order identifier
            - symbol (str): Stock symbol
            - entry_price (float): Entry price of the order
            - target (float): Target price
            - stoploss (float): Stop loss price
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    qty = params["quantity"]
    status.emit("Placing", f"Smart order: {qty} {symbol}...")

    result = client.post("/v2/place_smart_order", data={
        "symbol": symbol,
        "quantity": qty,
        "transaction_type": params.get("transaction_type", "BUY"),
        "target_percent": params.get("target_percent", 2.0),
        "stoploss_percent": params.get("stoploss_percent", 1.0),
        "broker": params.get("broker"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Smart order failed for {symbol}"))

    return tool_response(
        order_id=result.get("order_id"),
        symbol=symbol,
        entry_price=result.get("entry_price"),
        target=result.get("target"),
        stoploss=result.get("stoploss"),
    )


@async_tool_wrapper(required_params=["symbol"])
async def exit_position_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exit an open position (sell all holdings of a symbol).

    Args:
        params: Dictionary containing:
            - symbol (str, required): Symbol to exit
            - quantity (int, optional): Partial exit quantity (default: all)
            - broker (str, optional): Broker filter

    Returns:
        Dictionary with:
            - order_id (str): Exit order identifier
            - symbol (str): Symbol exited
            - quantity_exited (int): Number of shares sold
            - exit_price (float): Price at which position was exited
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    status.emit("Exiting", f"Exiting position: {symbol}...")

    result = client.post("/v2/exit_position", data={
        "symbol": symbol,
        "quantity": params.get("quantity"),
        "broker": params.get("broker"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to exit position for {symbol}"))

    return tool_response(
        order_id=result.get("order_id"),
        symbol=symbol,
        quantity_exited=result.get("quantity_exited"),
        exit_price=result.get("exit_price"),
    )


@async_tool_wrapper(required_params=["order_id"])
async def cancel_order_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cancel a pending order.

    Args:
        params: Dictionary containing:
            - order_id (str, required): Order ID to cancel
            - broker (str, optional): Broker filter

    Returns:
        Dictionary with:
            - order_id (str): Cancelled order ID
            - cancelled (bool): Whether cancellation succeeded
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    order_id = params["order_id"]
    status.emit("Cancelling", f"Cancelling order {order_id}...")

    result = client.post("/v2/cancel_order", data={
        "order_id": order_id,
        "broker": params.get("broker"),
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to cancel order {order_id}"))

    return tool_response(
        order_id=order_id,
        cancelled=True,
    )


@async_tool_wrapper()
async def get_orders_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get order history with optional filters.

    Args:
        params: Dictionary containing:
            - status (str, optional): Filter by status (open, completed, cancelled)
            - broker (str, optional): Filter by broker
            - symbol (str, optional): Filter by symbol
            - limit (int, optional): Max results (default 50)

    Returns:
        Dictionary with:
            - orders (list): Order objects with order_id, symbol, quantity, status, price
            - count (int): Number of orders returned
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Loading order history...")

    result = client.get("/v2/orders", params={
        "status": params.get("status", ""),
        "broker": params.get("broker", ""),
        "symbol": params.get("symbol", ""),
        "limit": params.get("limit", 50),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get orders"))

    orders = result.get("orders", result.get("data", []))
    return tool_response(orders=orders, count=len(orders))


__all__ = [
    "place_order_tool",
    "place_smart_order_tool",
    "exit_position_tool",
    "cancel_order_tool",
    "get_orders_tool",
]
