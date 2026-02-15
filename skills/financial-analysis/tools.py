"""
Financial Analysis Skill
========================

LLM-powered financial analysis tools that fetch data from PlanMyInvesting
and use Claude/DSPy for intelligent analysis, sentiment, and comparisons.
"""

import json
import logging
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

load_jotty_env()
logger = logging.getLogger(__name__)
status = SkillStatus("financial-analysis")


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


def _get_lm():
    """Get the configured DSPy language model."""
    try:
        import dspy

        return dspy.settings.lm
    except Exception:
        return None


def _call_lm(prompt: str) -> str:
    """Call the LLM with a prompt and return text response."""
    lm = _get_lm()
    if not lm:
        return "[LLM not available - configure DSPy language model]"
    try:
        result = lm(prompt)
        if isinstance(result, list) and result:
            return result[0] if isinstance(result[0], str) else str(result[0])
        return str(result)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"[LLM error: {e}]"


@async_tool_wrapper(required_params=["symbol"])
async def sentiment_analysis_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze market sentiment for a stock using recent data and LLM analysis.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol to analyze
            - include_news (bool, optional): Include news-based analysis (default True)

    Returns:
        Dictionary with sentiment score, analysis narrative, signals
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    status.emit("Fetching", f"Getting data for {symbol}...")

    # Fetch quote and chart data
    quote = client.get("/v2/get_ltp", params={"symbol": symbol})
    chart = client.get(
        "/v2/get_chart_data",
        params={
            "symbol": symbol,
            "interval": "1d",
            "days": 14,
        },
    )

    quote_data = (
        {
            "ltp": quote.get("ltp"),
            "change": quote.get("change"),
            "change_percent": quote.get("change_percent"),
            "volume": quote.get("volume"),
        }
        if quote.get("success")
        else {}
    )

    candles = chart.get("candles", chart.get("data", [])) if chart.get("success") else []

    status.emit("Analyzing", f"Running sentiment analysis for {symbol}...")

    prompt = f"""Analyze the market sentiment for {symbol} based on this data:

Current Quote: {json.dumps(quote_data, default=str)}
Last 14 days OHLCV (recent first): {json.dumps(candles[:14], default=str)}

Provide:
1. Overall sentiment: BULLISH, BEARISH, or NEUTRAL
2. Confidence level: HIGH, MEDIUM, or LOW
3. Key signals (3-5 bullet points)
4. Short narrative (2-3 sentences)

Format as JSON with keys: sentiment, confidence, signals (list), narrative"""

    analysis = _call_lm(prompt)

    # Try to parse LLM response as JSON
    try:
        parsed = json.loads(analysis, strict=False)
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw_analysis": analysis}

    return tool_response(
        symbol=symbol,
        sentiment=parsed.get("sentiment", "UNKNOWN"),
        confidence=parsed.get("confidence", "UNKNOWN"),
        signals=parsed.get("signals", []),
        narrative=parsed.get("narrative", parsed.get("raw_analysis", "")),
        quote=quote_data,
    )


@async_tool_wrapper(required_params=["symbol"])
async def earnings_analysis_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze earnings and financial performance of a stock.

    Args:
        params: Dictionary containing:
            - symbol (str, required): Stock symbol to analyze
            - quarters (int, optional): Number of quarters to analyze (default 4)

    Returns:
        Dictionary with earnings analysis, trends, and recommendation
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbol = params["symbol"]
    quarters = params.get("quarters", 4)
    status.emit("Fetching", f"Getting financials for {symbol}...")

    # Fetch financials data
    financials = client.get(
        "/v2/get_financials",
        params={
            "symbol": symbol,
            "quarters": quarters,
        },
    )

    financials_data = financials.get("data", financials) if financials.get("success") else {}

    # Also get current quote for context
    quote = client.get("/v2/get_ltp", params={"symbol": symbol})
    quote_data = (
        {
            "ltp": quote.get("ltp"),
            "change_percent": quote.get("change_percent"),
        }
        if quote.get("success")
        else {}
    )

    status.emit("Analyzing", f"Analyzing earnings for {symbol}...")

    prompt = f"""Analyze the earnings and financial performance of {symbol}:

Financial Data ({quarters} quarters): {json.dumps(financials_data, default=str)}
Current Price: {json.dumps(quote_data, default=str)}

Provide:
1. Revenue trend: GROWING, STABLE, or DECLINING
2. Profit trend: GROWING, STABLE, or DECLINING
3. Key financial highlights (3-5 points)
4. Earnings quality assessment
5. Forward outlook (2-3 sentences)

Format as JSON with keys: revenue_trend, profit_trend, highlights (list), quality, outlook"""

    analysis = _call_lm(prompt)

    try:
        parsed = json.loads(analysis, strict=False)
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw_analysis": analysis}

    return tool_response(
        symbol=symbol,
        revenue_trend=parsed.get("revenue_trend", "UNKNOWN"),
        profit_trend=parsed.get("profit_trend", "UNKNOWN"),
        highlights=parsed.get("highlights", []),
        quality=parsed.get("quality", ""),
        outlook=parsed.get("outlook", parsed.get("raw_analysis", "")),
        quote=quote_data,
    )


@async_tool_wrapper(required_params=["symbols"])
async def stock_comparison_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare multiple stocks with LLM-generated narrative analysis.

    Args:
        params: Dictionary containing:
            - symbols (list[str], required): List of symbols to compare (2-5)
            - criteria (str, optional): Comparison focus (e.g. "value", "growth", "momentum")

    Returns:
        Dictionary with comparison table, narrative, and ranking
    """
    status.set_callback(params.pop("_status_callback", None))
    client = _get_pmi_client()
    err = _require_client(client)
    if err:
        return err

    symbols = params["symbols"]
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    if len(symbols) < 2:
        return tool_error("Need at least 2 symbols to compare")
    if len(symbols) > 5:
        symbols = symbols[:5]

    criteria = params.get("criteria", "overall")
    status.emit("Fetching", f"Getting data for {len(symbols)} stocks...")

    # Fetch quotes for all symbols
    quotes_data = {}
    for symbol in symbols:
        quote = client.get("/v2/get_ltp", params={"symbol": symbol})
        if quote.get("success"):
            quotes_data[symbol] = {
                "ltp": quote.get("ltp"),
                "change": quote.get("change"),
                "change_percent": quote.get("change_percent"),
                "volume": quote.get("volume"),
            }
        else:
            quotes_data[symbol] = {"error": quote.get("error", "Data unavailable")}

    status.emit("Analyzing", "Generating comparison analysis...")

    prompt = f"""Compare these stocks based on {criteria} criteria:

Stocks and Data: {json.dumps(quotes_data, default=str)}

Provide:
1. Ranking from best to worst with brief reason for each
2. Key differentiators (3-5 points)
3. Recommendation narrative (3-4 sentences)
4. Risk comparison

Format as JSON with keys: ranking (list of {{symbol, rank, reason}}), differentiators (list), narrative, risk_comparison"""

    analysis = _call_lm(prompt)

    try:
        parsed = json.loads(analysis, strict=False)
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw_analysis": analysis}

    return tool_response(
        symbols=symbols,
        criteria=criteria,
        ranking=parsed.get("ranking", []),
        differentiators=parsed.get("differentiators", []),
        narrative=parsed.get("narrative", parsed.get("raw_analysis", "")),
        risk_comparison=parsed.get("risk_comparison", ""),
        quotes=quotes_data,
    )


__all__ = [
    "sentiment_analysis_tool",
    "earnings_analysis_tool",
    "stock_comparison_tool",
]
