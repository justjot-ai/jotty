---
name: analyzing-financials
description: "AI-driven financial analysis tools that combine real market data from PlanMyInvesting with LLM intelligence for sentiment analysis, earnings analysis, and multi-stock comparisons. Use when the user wants to stock, financial, analyze."
---

# Financial Analysis

LLM-powered financial analysis using PlanMyInvesting data.

## Description

AI-driven financial analysis tools that combine real market data from PlanMyInvesting with LLM intelligence for sentiment analysis, earnings analysis, and multi-stock comparisons.

## Type
derived

## Base Skills
- pmi-market-data
- claude-cli-llm

## Capabilities
- finance
- analyze
- research


## Triggers
- "stock"
- "financial"
- "analyze"
- "investment"
- "stock price"
- "financial data"
- "analyze stock"
- "market data"
- "portfolio"

## Category
workflow-automation

## Use When
User wants sentiment analysis, earnings analysis, or stock comparison with AI-generated insights

## Tools

### sentiment_analysis_tool
Analyze market sentiment for a stock using data and LLM.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `include_news` (bool, optional): Include news analysis

**Returns:**
- `sentiment` (BULLISH/BEARISH/NEUTRAL), `confidence`, `signals`, `narrative`

### earnings_analysis_tool
Analyze earnings and financial performance of a stock.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `quarters` (int, optional): Quarters to analyze (default 4)

**Returns:**
- `revenue_trend`, `profit_trend`, `highlights`, `quality`, `outlook`

### stock_comparison_tool
Compare multiple stocks with AI narrative.

**Parameters:**
- `symbols` (list[str], required): Symbols to compare (2-5)
- `criteria` (str, optional): Focus (value, growth, momentum)

**Returns:**
- `ranking`, `differentiators`, `narrative`, `risk_comparison`
