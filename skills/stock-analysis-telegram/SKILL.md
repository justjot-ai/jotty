---
name: stockanalysistelegram
description: "Fetch stock data, analyze with AI, create chart, and send to Telegram. Use when user wants stock analysis report."
---

# Stock Analysis to Telegram Skill

## Description
Composite skill that performs complete stock analysis workflow: data fetch → AI analysis → chart generation → Telegram delivery. Consolidates 4 separate tools into one operation.

## Type
composite

## Capabilities
- data-fetch
- analyze
- generate
- communicate

## Triggers
- "analyze stock and send"
- "stock report to telegram"
- "get stock analysis"
- "analyze [TICKER] stock"

## Category
data-analysis

## Base Skills
- stock-data-fetcher (or web-search for stock data)
- claude-cli-llm
- chart-generator
- telegram-sender

## Execution Mode
sequential

## Tools

### stock_analysis_telegram_tool
Fetch stock data, analyze performance, create chart, send via Telegram (all-in-one).

**Parameters:**
- `ticker` (str, required): Stock ticker symbol (e.g., "AAPL", "TSLA")
- `period` (str, optional): Analysis period - "1d", "5d", "1mo", "3mo", "6mo", "1y". Defaults to "1mo"
- `telegram_chat_id` (str, required): Telegram chat ID to send report
- `include_chart` (bool, optional): Include price chart. Defaults to True

**Returns:**
- `success` (bool): Whether operation completed successfully
- `ticker` (str): Stock ticker analyzed
- `current_price` (float): Current stock price
- `price_change` (float): Price change percentage
- `analysis` (str): AI-generated analysis summary
- `chart_path` (str, optional): Path to generated chart
- `telegram_sent` (bool): Whether report was sent
- `error` (str, optional): Error message if failed

## Usage Examples
```python
# Example 1: Monthly AAPL analysis
result = stock_analysis_telegram_tool({
    'ticker': 'AAPL',
    'telegram_chat_id': '123456789'
})

# Example 2: Yearly TSLA analysis with chart
result = stock_analysis_telegram_tool({
    'ticker': 'TSLA',
    'period': '1y',
    'telegram_chat_id': '123456789',
    'include_chart': True
})
```

## Requirements
- Stock data API (Yahoo Finance, Alpha Vantage, or similar)
- claude-cli-llm skill
- chart-generator skill (matplotlib-based)
- telegram-sender skill
- TELEGRAM_TOKEN environment variable

## Workflow
1. **Fetch**: Get stock data for period
2. **Analyze**: AI analyzes performance, trends, key metrics
3. **Visualize**: Generate price chart (optional)
4. **Deliver**: Send analysis + chart to Telegram

## Error Handling
Common errors and solutions:
- **Invalid ticker**: Use valid stock symbol. Example: "AAPL" not "Apple"
- **Data fetch failed**: Check stock data API key and internet connection
- **Chart generation failed**: Verify matplotlib is installed: `pip install matplotlib`
- **Telegram send failed**: Check TELEGRAM_TOKEN and chat_id are correct
