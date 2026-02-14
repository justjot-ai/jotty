# Web Search Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`search_web_tool`](#search_web_tool) | Search the web using Google (Serper API), SearXNG, or DuckDuckGo. |
| [`fetch_webpage_tool`](#fetch_webpage_tool) | Fetch and extract text content from a web page. |
| [`search_and_scrape_tool`](#search_and_scrape_tool) | Search web and scrape content from top results. |
| [`serper_scrape_website_tool`](#serper_scrape_website_tool) | Scrape a website using Serper API for clean markdown extraction. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get`](#get) | Return cached result if present and not expired, else None. |
| [`set`](#set) | Store a result with current timestamp. |
| [`clear`](#clear) | Clear all cached entries. |

---

## `search_web_tool`

Search the web using Google (Serper API), SearXNG, or DuckDuckGo.  Priority: Serper API (Google) > SearXNG > DuckDuckGo library > HTML parsing fallback.

**Parameters:**

- **query** (`str, required`): Search query
- **max_results** (`int, optional`): Max results (default: 10, max: 20)
- **provider** (`str, optional`): 'serper', 'searxng', or 'duckduckgo' (default: auto)

**Returns:** Dictionary with success, results, count, query, provider

---

## `fetch_webpage_tool`

Fetch and extract text content from a web page.

**Parameters:**

- **url** (`str, required`): URL to fetch
- **max_length** (`int, optional`): Max content length (default: 10000)
- **max_retries** (`int, optional`): Max retry attempts (default: 3)

**Returns:** Dictionary with success, url, title, content, length

---

## `search_and_scrape_tool`

Search web and scrape content from top results.

**Parameters:**

- **query** (`str, required`): Search query
- **num_results** (`int, optional`): Number of search results (default: 5)
- **scrape_top** (`int, optional`): Number of results to scrape (default: 3)
- **max_content_length** (`int, optional`): Max content per page (default: 5000)

**Returns:** Dictionary with success, query, results, count, scraped_count, provider

---

## `serper_scrape_website_tool`

Scrape a website using Serper API for clean markdown extraction.  Unlike fetch_webpage_tool which does basic HTML scraping, this uses Serper's dedicated scraping endpoint for higher quality markdown output.

**Parameters:**

- **url** (`str, required`): URL to scrape
- **max_length** (`int, optional`): Max content length (default: 100000)

**Returns:** Dictionary with success, url, title, content, content_length

---

## `get`

Return cached result if present and not expired, else None.

**Parameters:**

- **key** (`str`)

**Returns:** `Optional[Any]`

---

## `set`

Store a result with current timestamp.

**Parameters:**

- **key** (`str`)
- **value** (`Any`)

**Returns:** `None`

---

## `clear`

Clear all cached entries.

**Returns:** `None`
