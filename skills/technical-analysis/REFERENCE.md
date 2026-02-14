# technical-analysis - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`technical_analysis_tool`](#technical_analysis_tool) | Run multi-timeframe technical analysis on a stock. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`load_ohlcv`](#load_ohlcv) | Load OHLCV data from NSE compressed CSV files. |
| [`add_all_indicators`](#add_all_indicators) | Add all technical indicators to the DataFrame. |
| [`generate_signals`](#generate_signals) | Generate trading signals from a DataFrame with indicators already added. |
| [`get_latest_indicators`](#get_latest_indicators) | Extract a dict of the most recent indicator values. |

---

## `technical_analysis_tool`

Run multi-timeframe technical analysis on a stock.  Parameters (via ``params`` dict): ticker (str): NSE stock symbol, e.g. 'RELIANCE'. timeframes (list[str], optional): Default ['60minute', 'Day']. data_path (str, optional): Override data directory. format (str, optional): 'json' (default) or 'summary'.

**Parameters:**

- **params** (`dict`)

**Returns:** dict with ``success``, ``data`` (timeframes, signals, support/resistance, trend, indicators).

---

## `load_ohlcv`

Load OHLCV data from NSE compressed CSV files.  Returns a DataFrame indexed by date with columns: open, high, low, close, volume. Returns ``None`` when no data can be found/loaded.

**Parameters:**

- **ticker** (`str`)
- **timeframe** (`str`)
- **data_path** (`str`)

**Returns:** `Optional['pd.DataFrame']`

---

## `add_all_indicators`

Add all technical indicators to the DataFrame.  Tries pandas_ta first, falls back to ta library.

**Parameters:**

- **df** (`'pd.DataFrame'`)

**Returns:** `'pd.DataFrame'`

---

## `generate_signals`

Generate trading signals from a DataFrame with indicators already added.

**Parameters:**

- **df** (`'pd.DataFrame'`)
- **timeframe** (`str`)

**Returns:** `Dict[str, Any]`

---

## `get_latest_indicators`

Extract a dict of the most recent indicator values.

**Parameters:**

- **df** (`'pd.DataFrame'`)

**Returns:** `Dict[str, Any]`
