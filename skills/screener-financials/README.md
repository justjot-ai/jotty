# Screener.in Financials Skill

Fetches financial data for Indian companies from screener.in with automatic proxy rotation to avoid blocking.

## Features

✅ **Free Proxy Rotation** - Automatically rotates through free proxies  
✅ **Rate Limiting Protection** - Built-in retry logic with exponential backoff  
✅ **User-Agent Rotation** - Randomizes user agents to avoid detection  
✅ **Multiple Data Types** - P&L, Balance Sheet, Cash Flow, Ratios  
✅ **Multiple Formats** - JSON, Markdown, CSV output  
✅ **Company Search** - Search by company name  

## Usage Examples

### Search for a Company

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('screener-financials')
result = await skill.tools['search_company_tool']({
    'query': 'Reliance',
    'max_results': 5
})

print(result['results'])
```

### Get Company Financials

```python
# Get all financial data
result = await skill.tools['get_company_financials_tool']({
    'company_name': 'RELIANCE',  # or company code
    'data_type': 'all',  # 'all', 'pl', 'balance_sheet', 'cash_flow', 'ratios'
    'period': 'annual',  # 'annual' or 'quarterly'
    'format': 'json',  # 'json', 'markdown', 'csv'
    'use_proxy': True,  # Enable proxy rotation
    'max_retries': 3
})

print(result['data'])
```

### Get Only Ratios

```python
result = await skill.tools['get_company_ratios_tool']({
    'company_name': 'RELIANCE',
    'period': 'annual'
})

print(result['ratios'])
```

## Proxy Rotation

The skill automatically:
1. Fetches free proxies from multiple sources
2. Rotates proxies on each request
3. Marks failed proxies and skips them
4. Falls back to direct connection if no proxies available
5. Uses exponential backoff on failures

### Free Proxy Sources Used:
- ProxyScrape API
- Proxy-List.download API
- Geonode API (free tier)

## Rate Limiting

- Automatic retry with exponential backoff
- Configurable max retries (default: 3)
- Random delays between requests
- User-agent rotation

## Output Formats

### JSON (Default)
```json
{
  "success": true,
  "company_name": "Reliance Industries",
  "company_code": "RELIANCE",
  "period": "annual",
  "data": {
    "profit_loss": {...},
    "balance_sheet": {...},
    "cash_flow": {...},
    "ratios": {...}
  }
}
```

### Markdown
Formatted tables and sections for easy reading.

### CSV
Comma-separated values for spreadsheet import.

## Error Handling

The skill handles:
- ✅ Proxy failures (automatic rotation)
- ✅ Rate limiting (exponential backoff)
- ✅ Blocked requests (proxy rotation + retry)
- ✅ Network errors (retry logic)
- ✅ Company not found (search fallback)

## Dependencies

- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `lxml` - Fast HTML parser

Install:
```bash
pip install requests beautifulsoup4 lxml
```

## Notes

- Free proxies may be slow or unreliable
- For production use, consider paid proxy services
- Screener.in may change HTML structure (parsing may need updates)
- Respect screener.in's terms of service and rate limits
