# Web Scraper Skill

## Description
Advanced web scraping capabilities using Scrapy. Supports single page scraping or spider mode (following links), with proxy support for geo-blocked content.


## Type
base

## Tools

### scrape_website_tool
Scrapes a website and extracts content.

**Parameters:**
- `url` (str, required): URL to scrape
- `follow_links` (bool, optional): Follow links on the page (spider mode), default: False
- `max_pages` (int, optional): Maximum pages to scrape in spider mode, default: 10
- `output_format` (str, optional): Output format - 'markdown', 'html', 'text', default: 'markdown'
- `selectors` (dict, optional): Custom CSS selectors for title, content, author, date
- `exclude_patterns` (list, optional): URL patterns to exclude
- `proxy` (str, optional): Proxy URL for geo-blocked sites
- `timeout` (int, optional): Request timeout in seconds, default: 30

**Returns:**
- `success` (bool): Whether scraping succeeded
- `url` (str): Scraped URL
- `title` (str): Page title
- `content` (str): Extracted content
- `pages_scraped` (int): Number of pages scraped
- `error` (str, optional): Error message if failed
