"""
Stock Analysis to Telegram Skill - Composite Workflow

Consolidates: data fetch → AI analysis → chart → Telegram

Follows Anthropic best practices:
- Clear error messages with corrective examples
- Semantic response fields
- Status reporting
- Proper validation
"""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

status = SkillStatus("stock-analysis-telegram")


@async_tool_wrapper(required_params=["ticker", "telegram_chat_id"])
async def stock_analysis_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch stock data, analyze, chart, and send to Telegram (all-in-one).

    Composite workflow:
    1. Fetch stock data for period
    2. AI analysis of performance
    3. Generate price chart (optional)
    4. Send to Telegram

    Args:
        params: Dictionary containing:
            - ticker (str, required): Stock ticker (e.g., "AAPL")
            - period (str, optional): "1d", "5d", "1mo", "3mo", "6mo", "1y". Default: "1mo"
            - telegram_chat_id (str, required): Telegram chat ID
            - include_chart (bool, optional): Include chart. Default: True

    Returns:
        Dictionary with success, ticker, current_price, price_change, analysis,
        chart_path, telegram_sent, error
    """
    status.set_callback(params.pop("_status_callback", None))

    ticker = params.get("ticker", "").upper()
    period = params.get("period", "1mo")
    telegram_chat_id = params.get("telegram_chat_id")
    include_chart = params.get("include_chart", True)

    # Validate ticker
    if not ticker or len(ticker) > 5:
        return tool_error(
            f'Invalid ticker symbol: "{ticker}". Use valid stock symbol. '
            f'Examples: "AAPL", "TSLA", "GOOGL", "MSFT"'
        )

    # Validate period
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
    if period not in valid_periods:
        return tool_error(
            f'Invalid period: "{period}". Must be one of: {", ".join(valid_periods)}. '
            f'Example: {{"period": "1mo"}}'
        )

    from Jotty.core.capabilities.registry import get_unified_registry

    registry = get_unified_registry()

    try:
        # Step 1: Fetch stock data
        status.emit("Fetching", f"📈 Fetching {ticker} data ({period})...")

        # Try stock-data-fetcher skill, fallback to web search
        stock_data = None
        stock_skill = registry.get_skill("stock-data-fetcher")

        if stock_skill:
            data_tool = stock_skill.get_tool("fetch_stock_data_tool")
            data_result = await data_tool({"ticker": ticker, "period": period})

            if data_result.get("success"):
                stock_data = data_result
        else:
            # Fallback: web search for stock data
            search_skill = registry.get_skill("web-search")
            if not search_skill:
                return tool_error(
                    "No stock data source available. Install stock-data-fetcher or web-search skill."
                )

            search_tool = search_skill.get_tool("web_search_tool")
            search_result = await search_tool(
                {"query": f"{ticker} stock price {period}", "limit": 5}
            )

            if search_result.get("success"):
                # Parse search results for stock data
                stock_data = {
                    "ticker": ticker,
                    "current_price": "N/A",
                    "price_change": "N/A",
                    "data_source": "web_search",
                }

        if not stock_data:
            return tool_error(
                f"Failed to fetch data for {ticker}. "
                f"Verify ticker is valid. Check https://finance.yahoo.com/quote/{ticker}"
            )

        current_price = stock_data.get("current_price", "N/A")
        price_change = stock_data.get("price_change", 0.0)

        # Step 2: AI Analysis
        status.emit("Analyzing", f"🤖 Analyzing {ticker} performance...")

        llm_skill = registry.get_skill("claude-cli-llm")
        if not llm_skill:
            return tool_error(
                "LLM skill not available. Install claude-cli-llm skill. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY."
            )

        llm_tool = llm_skill.get_tool("claude_cli_llm_tool")

        analysis_prompt = f"""Analyze {ticker} stock performance over {period}:

Current Price: {current_price}
Change: {price_change}%

Provide:
1. Performance Summary (1-2 sentences)
2. Key Trends
3. Risk Level (Low/Medium/High)
4. Recommendation (Hold/Buy/Sell - for educational purposes)

Keep analysis concise (under 200 words)."""

        analysis_result = await llm_tool({"prompt": analysis_prompt, "max_tokens": 500})

        if not analysis_result.get("success"):
            return tool_error(
                f'AI analysis failed: {analysis_result.get("error")}. '
                f"Check LLM provider configuration."
            )

        analysis_text = analysis_result.get("response", "Analysis unavailable")

        # Step 3: Generate chart (optional)
        chart_path = None
        if include_chart:
            status.emit("Charting", f"📊 Creating {ticker} chart...")

            chart_skill = registry.get_skill("chart-generator")
            if chart_skill:
                chart_tool = chart_skill.get_tool("create_chart_tool")
                chart_result = await chart_tool(
                    {
                        "data": stock_data.get("price_history", []),
                        "chart_type": "line",
                        "title": f"{ticker} Stock Price - {period}",
                        "xlabel": "Date",
                        "ylabel": "Price (USD)",
                    }
                )

                if chart_result.get("success"):
                    chart_path = chart_result.get("path")

        # Step 4: Send to Telegram
        status.emit("Sending", f"📤 Sending {ticker} report to Telegram...")

        telegram_skill = registry.get_skill("telegram-sender")
        if not telegram_skill:
            return tool_error(
                "Telegram sender skill not available. "
                "Install telegram-sender and set TELEGRAM_TOKEN environment variable."
            )

        telegram_tool = telegram_skill.get_tool("telegram_send_tool")

        # Format message
        message = f"""📈 **{ticker} Stock Analysis**

**Period**: {period}
**Current Price**: ${current_price}
**Change**: {price_change:+.2f}%

**Analysis:**
{analysis_text}

---
_Generated by Jotty AI_"""

        # Send with or without chart
        telegram_params = {"chat_id": telegram_chat_id, "message": message}

        if chart_path:
            telegram_params["file"] = chart_path

        telegram_result = await telegram_tool(telegram_params)

        if not telegram_result.get("success"):
            return tool_error(
                f'Telegram send failed: {telegram_result.get("error")}. '
                f'Verify TELEGRAM_TOKEN and chat_id: "{telegram_chat_id}"'
            )

        status.emit("Complete", f"✅ {ticker} analysis sent!")

        return tool_response(
            ticker=ticker,
            current_price=current_price,
            price_change=price_change,
            analysis=analysis_text,
            chart_path=chart_path,
            telegram_sent=True,
            period=period,
        )

    except Exception as e:
        return tool_error(
            f"Stock analysis workflow failed: {str(e)}. "
            f"Verify all required skills are installed and configured."
        )


__all__ = ["stock_analysis_telegram_tool"]
