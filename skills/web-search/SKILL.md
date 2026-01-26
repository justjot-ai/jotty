# Web Search Skill

## Description
Provides web search capabilities and web page content retrieval using DuckDuckGo (no API key required).

## Tools

### search_web_tool
Searches the web using DuckDuckGo.

**Parameters:**
- `query` (str, required): Search query
- `max_results` (int, optional): Maximum number of results (default: 10, max: 20)

**Returns:**
- `success` (bool): Whether search succeeded
- `results` (list): List of search results with title, url, snippet
- `count` (int): Number of results
- `error` (str, optional): Error message if failed

### fetch_webpage_tool
Fetches and extracts text content from a web page.

**Parameters:**
- `url` (str, required): URL to fetch
- `max_length` (int, optional): Maximum length of extracted text (default: 10000)

**Returns:**
- `success` (bool): Whether fetch succeeded
- `url` (str): Fetched URL
- `title` (str): Page title
- `content` (str): Extracted text content
- `length` (int): Length of content
- `error` (str, optional): Error message if failed
