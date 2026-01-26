# Investing.com Commodities Price Fetcher

Fetches latest commodities prices from investing.com.

## Tools

### `get_commodities_prices_tool`

Fetches latest commodities prices from investing.com.

**Parameters:**
- `category` (str, optional): Category filter - 'energy', 'metals', 'agriculture', or 'all' (default: 'all')
- `format` (str, optional): Output format - 'json', 'markdown', 'text' (default: 'markdown')

**Returns:**
- `success` (bool): Whether fetch succeeded
- `commodities` (list): List of commodities with prices
- `formatted_output` (str): Formatted output string
- `timestamp` (str): Fetch timestamp
- `error` (str, optional): Error message if failed

**Example:**
```python
result = get_commodities_prices_tool({
    'category': 'metals',
    'format': 'markdown'
})
```
