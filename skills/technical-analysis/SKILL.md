---
name: analyzing-technicals
description: "Multi-timeframe technical analysis for NSE stocks — 30+ indicators, buy/sell signals, support/resistance."
---

# technical-analysis

Multi-timeframe technical analysis for NSE stocks — 30+ indicators, buy/sell signals, support/resistance.

## Type
base

## Capabilities
- analyze
- data-fetch

## Use When
User wants technical analysis, stock indicators, buy/sell signals, or chart patterns

## Tools

### `technical_analysis_tool`

Analyze a stock ticker across multiple timeframes. Calculates 30+ indicators
(trend, momentum, volatility, volume, overlap) and generates buy/sell signals.

**Parameters:**
- `ticker` (str, required): NSE stock symbol (e.g., 'RELIANCE')
- `timeframes` (list[str], optional): Timeframes to analyze. Default: ['60minute', 'Day']
  Supported: 15minute, 30minute, 60minute, day, week
- `data_path` (str, optional): Path to NSE data directory
- `format` (str, optional): 'json' (default) or 'summary'

**Returns:**
- `success` (bool)
- `data`: Dict with timeframes, signals, support/resistance, overall trend, indicators

## Dependencies

- pandas
- pandas_ta (preferred) or ta (fallback)

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Triggers
- "technical analysis"

## Category
workflow-automation
