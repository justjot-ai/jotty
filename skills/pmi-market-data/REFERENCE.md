# PMI Market Data - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`get_quote_tool`](#get_quote_tool) | Get live quote (LTP, change, volume) for a single symbol. |
| [`get_quotes_tool`](#get_quotes_tool) | Get live quotes for multiple symbols in one call. |
| [`search_symbols_tool`](#search_symbols_tool) | Search for stock symbols by name or keyword. |
| [`get_indices_tool`](#get_indices_tool) | Get major market indices (NIFTY 50, SENSEX, BANK NIFTY, etc. |
| [`get_chart_data_tool`](#get_chart_data_tool) | Get OHLCV chart data for a symbol. |
| [`get_market_breadth_tool`](#get_market_breadth_tool) | Get market breadth data (advances, declines, unchanged). |
| [`get_sector_analysis_tool`](#get_sector_analysis_tool) | Get sector-wise performance analysis. |

---

## `get_quote_tool`

Get live quote (LTP, change, volume) for a single symbol.

**Parameters:**

- **symbol** (`str, required`): Stock symbol (e.g. "RELIANCE", "NIFTY 50")

**Returns:** Dictionary with: - symbol (str): Stock symbol - ltp (float): Last traded price - change (float): Absolute price change - change_percent (float): Percentage price change - volume (int): Trading volume - timestamp (str): Quote timestamp

---

## `get_quotes_tool`

Get live quotes for multiple symbols in one call.

**Parameters:**

- **symbols** (`list, required`): List of stock symbols

**Returns:** Dictionary with: - quotes (list): List of quote objects with ltp, change, volume per symbol - count (int): Number of quotes returned

---

## `search_symbols_tool`

Search for stock symbols by name or keyword.

**Parameters:**

- **query** (`str, required`): Search query (e.g. "reliance", "banking")
- **exchange** (`str, optional`): Filter by exchange (NSE, BSE)
- **limit** (`int, optional`): Max results (default 10)

**Returns:** Dictionary with: - symbols (list): Matching symbol objects with name, exchange, symbol - query (str): Original search query

---

## `get_indices_tool`

Get major market indices (NIFTY 50, SENSEX, BANK NIFTY, etc.).

**Parameters:**

- **exchange** (`str, optional`): Filter by exchange (NSE, BSE)

**Returns:** Dictionary with: - indices (list): List of index objects with name, value, change, change_percent

---

## `get_chart_data_tool`

Get OHLCV chart data for a symbol.

**Parameters:**

- **symbol** (`str, required`): Stock symbol
- **interval** (`str, optional`): Candle interval (1m, 5m, 15m, 1h, 1d) default "1d"
- **days** (`int, optional`): Number of days of history (default 30)

**Returns:** Dictionary with: - symbol (str): Stock symbol - interval (str): Candle interval used - candles (list): OHLCV candle data with open, high, low, close, volume - count (int): Number of candles returned

---

## `get_market_breadth_tool`

Get market breadth data (advances, declines, unchanged).

**Parameters:**

- **exchange** (`str, optional`): Exchange filter (NSE, BSE)

**Returns:** Dictionary with: - indices (list): Index-wise breadth data - category (str): Market category - advances (int): Number of advancing stocks - declines (int): Number of declining stocks - unchanged (int): Number of unchanged stocks

---

## `get_sector_analysis_tool`

Get sector-wise performance analysis.

**Parameters:**

- **period** (`str, optional`): Analysis period (1d, 1w, 1m) default "1d"

**Returns:** Dictionary with: - sectors (list): Sector objects with name, change_percent, top_gainers, top_losers - period (str): Analysis period used
