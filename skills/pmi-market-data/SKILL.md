# PMI Market Data

Real-time and historical market data from PlanMyInvesting.

## Description

Provides live quotes, multi-symbol quotes, symbol search, market indices, OHLCV chart data, market breadth, and sector analysis via the PlanMyInvesting REST API.

## Type
base

## Capabilities
- finance
- data-fetch

## Use When
User wants to get stock quotes, search symbols, view market indices, get chart data, check market breadth, or see sector performance

## Tools

### get_quote_tool
Get live quote for a single symbol.

**Parameters:**
- `symbol` (str, required): Stock symbol (e.g. "RELIANCE", "NIFTY 50")

**Returns:**
- `symbol`, `ltp`, `change`, `change_percent`, `volume`, `timestamp`

### get_quotes_tool
Get live quotes for multiple symbols in one call.

**Parameters:**
- `symbols` (list[str], required): List of stock symbols

**Returns:**
- `quotes` (list), `count` (int)

### search_symbols_tool
Search for stock symbols by name or keyword.

**Parameters:**
- `query` (str, required): Search query
- `exchange` (str, optional): Filter by exchange (NSE, BSE)
- `limit` (int, optional): Max results (default 10)

**Returns:**
- `symbols` (list), `query` (str)

### get_indices_tool
Get major market indices.

**Parameters:**
- `exchange` (str, optional): Filter by exchange

**Returns:**
- `indices` (list)

### get_chart_data_tool
Get OHLCV chart data for a symbol.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `interval` (str, optional): Candle interval (1m, 5m, 15m, 1h, 1d)
- `days` (int, optional): Days of history (default 30)

**Returns:**
- `symbol`, `interval`, `candles` (list), `count`

### get_market_breadth_tool
Get market breadth data.

**Parameters:**
- `exchange` (str, optional): Exchange filter

**Returns:**
- `advances`, `declines`, `unchanged`

### get_sector_analysis_tool
Get sector-wise performance analysis.

**Parameters:**
- `period` (str, optional): Period (1d, 1w, 1m)

**Returns:**
- `sectors` (list), `period`
