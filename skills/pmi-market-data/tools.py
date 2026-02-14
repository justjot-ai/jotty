"""
PMI Market Data Skill
=====================

Real-time and historical market data from PlanMyInvesting API.
Provides quotes, indices, charts, breadth, and sector analysis.
"""

import logging
from typing import Dict, Any

from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("pmi-market-data")


def _get_pmi_client():
    """Lazy import to avoid circular deps with hyphenated directory."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pmi_client",
        Path(__file__).resolve().parent / "pmi_client.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlanMyInvestingClient()


def _require_client(client):
    """Return error dict if client is not configured."""
    err = client.require_token()
    if err:
        return err
    return None


@async_tool_wrapper(required_params=["symbol"])
async def get_quote_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get live quote (LTP, change, volume) for a single symbol.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol (e.g. "RELIANCE", "NIFTY 50")

    Returns:
        Dictionary with symbol, ltp, change, change_percent, volume
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    status.emit("Fetching", f"Getting quote for {symbol}...")

    result = client.get("/v2/get_ltp", params={"symbol": symbol})
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to get quote for {symbol}"))

    return tool_response(
        symbol=symbol,
        ltp=result.get("ltp"),
        change=result.get("change"),
        change_percent=result.get("change_percent"),
        volume=result.get("volume"),
        timestamp=result.get("timestamp"),
    )


@async_tool_wrapper(required_params=["symbols"])
async def get_quotes_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get live quotes for multiple symbols in one call.

    Args:
        params: Dictionary containing:
            - symbols (list[str], required): List of stock symbols

    Returns:
        Dictionary with quotes list and count
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbols = params["symbols"]
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    status.emit("Fetching", f"Getting quotes for {len(symbols)} symbols...")

    result = client.post("/v2/get_ltps", data={"symbols": symbols})
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get quotes"))

    return tool_response(
        quotes=result.get("quotes", result.get("data", [])),
        count=len(result.get("quotes", result.get("data", []))),
    )


@async_tool_wrapper(required_params=["query"])
async def search_symbols_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for stock symbols by name or keyword.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query (e.g. "reliance", "banking")
            - exchange (str, optional): Filter by exchange (NSE, BSE)
            - limit (int, optional): Max results (default 10)

    Returns:
        Dictionary with matching symbols list
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    query = params["query"]
    status.emit("Searching", f"Searching symbols: {query}...")

    result = client.get("/api/search/symbols", params={
        "q": query,
        "exchange": params.get("exchange", ""),
        "limit": params.get("limit", 10),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Symbol search failed"))

    return tool_response(
        symbols=result.get("results", result.get("symbols", result.get("data", []))),
        query=query,
    )


@async_tool_wrapper()
async def get_indices_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get major market indices (NIFTY 50, SENSEX, BANK NIFTY, etc.).

    Args:
        params: Dictionary containing:
            - exchange (str, optional): Filter by exchange (NSE, BSE)

    Returns:
        Dictionary with indices list
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Getting market indices...")

    result = client.get("/v2/get_indices", params={
        "exchange": params.get("exchange", ""),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get indices"))

    return tool_response(
        indices=result.get("indices", result.get("data", [])),
    )


@async_tool_wrapper(required_params=["symbol"])
async def get_chart_data_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get OHLCV chart data for a symbol.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol
            - interval (str, optional): Candle interval (1m, 5m, 15m, 1h, 1d) default "1d"
            - days (int, optional): Number of days of history (default 30)

    Returns:
        Dictionary with candles (OHLCV) list
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    interval = params.get("interval", "1d")
    days = params.get("days", 30)
    status.emit("Fetching", f"Getting chart data for {symbol} ({interval})...")

    result = client.get("/api/chart/data", params={
        "symbol": symbol,
        "interval": interval,
        "days": days,
    })
    if not result.get("success"):
        return tool_error(result.get("error", f"Failed to get chart data for {symbol}"))

    return tool_response(
        symbol=symbol,
        interval=interval,
        candles=result.get("candles", result.get("data", [])),
        count=len(result.get("candles", result.get("data", []))),
    )


@async_tool_wrapper()
async def get_market_breadth_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get market breadth data (advances, declines, unchanged).

    Args:
        params: Dictionary containing:
            - exchange (str, optional): Exchange filter (NSE, BSE)

    Returns:
        Dictionary with advances, declines, unchanged counts
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    status.emit("Fetching", "Getting market breadth...")

    result = client.get("/api/analysis/market-breadth", params={
        "exchange": params.get("exchange", ""),
    })
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get market breadth"))

    return tool_response(
        indices=result.get("indices"),
        category=result.get("category"),
        advances=result.get("advances"),
        declines=result.get("declines"),
        unchanged=result.get("unchanged"),
        data=result.get("data"),
    )


@async_tool_wrapper()
async def get_sector_analysis_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get sector-wise performance analysis.

    Args:
        params: Dictionary containing:
            - period (str, optional): Analysis period (1d, 1w, 1m) default "1d"

    Returns:
        Dictionary with sectors performance data
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    period = params.get("period", "1d")
    status.emit("Analyzing", f"Getting sector analysis ({period})...")

    result = client.get("/api/analysis/sectors", params={"period": period})
    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get sector analysis"))

    return tool_response(
        sectors=result.get("sectors", result.get("data", [])),
        period=period,
    )


__all__ = [
    "get_quote_tool",
    "get_quotes_tool",
    "search_symbols_tool",
    "get_indices_tool",
    "get_chart_data_tool",
    "get_market_breadth_tool",
    "get_sector_analysis_tool",
]
