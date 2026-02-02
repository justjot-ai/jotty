# Web Search Skill

Web search with Google (Serper API) or DuckDuckGo, plus content scraping.

## Description

Provides web search capabilities using Google (via Serper API) or DuckDuckGo, and web page content extraction. Supports combined search+scrape for deep research.

## Providers

### Serper API (Google Search) - Recommended
- Google-quality search results
- Requires `SERPER_API_KEY` environment variable
- Get API key at: https://serper.dev

### DuckDuckGo (Free, No API Key)
- Free alternative
- Uses duckduckgo-search library or HTML fallback
- Install: `pip install duckduckgo-search`

## Tools

### search_web_tool
Searches the web using Google (Serper) or DuckDuckGo.

**Parameters:**
- `query` (str, required): Search query
- `max_results` (int, optional): Maximum results (default: 10, max: 20)
- `provider` (str, optional): 'serper', 'duckduckgo', or 'auto' (default: auto)

**Returns:**
- `success` (bool): Whether search succeeded
- `results` (list): List of search results with title, url, snippet
- `count` (int): Number of results
- `provider` (str): Which provider was used
- `error` (str, optional): Error message if failed

### fetch_webpage_tool
Fetches and extracts text content from a web page.

**Parameters:**
- `url` (str, required): URL to fetch
- `max_length` (int, optional): Maximum text length (default: 10000)
- `max_retries` (int, optional): Retry attempts (default: 3)

**Returns:**
- `success` (bool): Whether fetch succeeded
- `url` (str): Fetched URL
- `title` (str): Page title
- `content` (str): Extracted text content
- `length` (int): Content length
- `error` (str, optional): Error message if failed

### search_and_scrape_tool
Search web and scrape content from top results.

**Parameters:**
- `query` (str, required): Search query
- `num_results` (int, optional): Search results to return (default: 5)
- `scrape_top` (int, optional): Results to scrape (default: 3)
- `max_content_length` (int, optional): Max content per page (default: 5000)

**Returns:**
- `success` (bool): Whether operation succeeded
- `query` (str): Search query
- `results` (list): Results with title, url, snippet, and scraped content
- `count` (int): Number of results
- `scraped_count` (int): Number of pages scraped
- `provider` (str): Search provider used
- `error` (str, optional): Error message if failed

## Environment Variables

- `SERPER_API_KEY`: Serper API key for Google search

## Usage

```python
# Basic search (auto-selects provider)
result = search_web_tool({'query': 'python tutorials'})

# Force specific provider
result = search_web_tool({
    'query': 'python tutorials',
    'provider': 'serper'
})

# Search and scrape for deep research
result = search_and_scrape_tool({
    'query': 'machine learning best practices',
    'scrape_top': 5
})
```

## Dependencies

- requests
- duckduckgo-search (optional, for better DuckDuckGo support)
