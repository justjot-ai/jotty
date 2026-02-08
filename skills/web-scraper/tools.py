"""
Web Scraper Skill

Scrape websites and extract content.
Refactored to use Jotty core utilities.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List
import html2text
from urllib.parse import urljoin, urlparse

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Browser-like headers
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def _scrape_single_page(
    url: str,
    output_format: str,
    selectors: Dict[str, str],
    headers: Dict[str, str],
    proxies: Optional[Dict[str, str]],
    timeout: int
) -> Dict[str, Any]:
    """Scrape a single page."""
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = ''
        if selectors.get('title'):
            title_elem = soup.select_one(selectors['title'])
            if title_elem:
                title = title_elem.get_text(strip=True)
        else:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

        # Extract content
        content = ''
        if selectors.get('content'):
            content_elem = soup.select_one(selectors['content'])
            if content_elem:
                content = content_elem.get_text(separator='\n', strip=True)
        else:
            main = soup.find('main') or soup.find('article') or soup.find('body')
            if main:
                for tag in main(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                content = main.get_text(separator='\n', strip=True)

        # Convert to requested format
        if output_format == 'markdown':
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.ignore_images = False
            h2t.body_width = 0
            content = h2t.handle(str(soup))
        elif output_format == 'html':
            content = str(soup)

        return tool_response(
            url=url,
            title=title,
            content=content[:100000],
            content_length=len(content),
            pages_scraped=1
        )

    except requests.RequestException as e:
        return tool_error(f'Network error: {str(e)}')


def _scrape_spider_mode(
    start_url: str,
    max_pages: int,
    output_format: str,
    selectors: Dict[str, str],
    exclude_patterns: List[str],
    headers: Dict[str, str],
    proxies: Optional[Dict[str, str]],
    timeout: int
) -> Dict[str, Any]:
    """Scrape multiple pages following links."""
    visited = set()
    to_visit = [start_url]
    pages_data = []
    domain = urlparse(start_url).netloc

    while to_visit and len(pages_data) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue

        if any(pattern in url for pattern in exclude_patterns):
            continue

        visited.add(url)

        page_result = _scrape_single_page(
            url, output_format, selectors, headers, proxies, timeout
        )

        if page_result.get('success'):
            pages_data.append(page_result)

            if len(pages_data) < max_pages:
                try:
                    response = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])
                        if urlparse(full_url).netloc == domain:
                            if full_url not in visited and full_url not in to_visit:
                                to_visit.append(full_url)
                except Exception:
                    pass

    combined_content = '\n\n---\n\n'.join([
        f"# {p.get('title', 'Untitled')}\n\n{p.get('content', '')}"
        for p in pages_data
    ])

    return tool_response(
        url=start_url,
        title=f'Scraped {len(pages_data)} pages',
        content=combined_content[:200000],
        content_length=len(combined_content),
        pages_scraped=len(pages_data),
        pages=[{'url': p['url'], 'title': p['title']} for p in pages_data]
    )


@tool_wrapper(required_params=['url'])
def scrape_website_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrape a website and extract content.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to scrape
            - follow_links (bool, optional): Spider mode (default: False)
            - max_pages (int, optional): Max pages in spider mode (default: 10)
            - output_format (str, optional): 'markdown', 'html', 'text' (default: 'markdown')
            - selectors (dict, optional): Custom CSS selectors
            - exclude_patterns (list, optional): URL patterns to exclude
            - proxy (str, optional): Proxy URL
            - timeout (int, optional): Request timeout (default: 30)

    Returns:
        Dictionary with success, url, title, content, pages_scraped
    """
    url = params['url']
    follow_links = params.get('follow_links', False)
    max_pages = params.get('max_pages', 10)
    output_format = params.get('output_format', 'markdown')
    selectors = params.get('selectors', {})
    exclude_patterns = params.get('exclude_patterns', [])
    proxy = params.get('proxy')
    timeout = params.get('timeout', 30)

    headers = HEADERS.copy()
    proxies = {'http': proxy, 'https': proxy} if proxy else None

    if follow_links:
        return _scrape_spider_mode(
            url, max_pages, output_format, selectors,
            exclude_patterns, headers, proxies, timeout
        )
    else:
        return _scrape_single_page(
            url, output_format, selectors, headers, proxies, timeout
        )


__all__ = ['scrape_website_tool']
