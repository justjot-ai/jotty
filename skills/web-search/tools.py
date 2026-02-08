"""
Web Search Skill

Search the web using Google (Serper API) or DuckDuckGo.
Refactored to use Jotty core utilities.
"""

import os
import re
import time
import logging
import requests
from typing import Dict, Any, List

from Jotty.core.utils.env_loader import load_jotty_env, get_env
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

load_jotty_env()

logger = logging.getLogger(__name__)

SERPER_API_KEY = get_env("SERPER_API_KEY")
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Try to load duckduckgo-search library
DDG_AVAILABLE = False
DDGS = None
try:
    try:
        from ddgs import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
except ImportError:
    logger.info("duckduckgo-search/ddgs library not available, using HTML parsing fallback")


def _serper_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Search using Serper API (Google Search)."""
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY not set")

    response = requests.post(
        'https://google.serper.dev/search',
        headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
        json={'q': query, 'num': num_results},
        timeout=10
    )
    response.raise_for_status()

    return [
        {
            'title': item.get('title', 'Untitled'),
            'url': item.get('link', ''),
            'snippet': item.get('snippet', '')
        }
        for item in response.json().get('organic', [])
    ]


def _ddg_search(query: str, max_results: int) -> List[Dict[str, str]]:
    """Search using DuckDuckGo library."""
    with DDGS() as ddgs:
        results_list = list(ddgs.text(query, max_results=max_results))

    return [
        {
            'title': item.get('title', 'Untitled'),
            'url': item.get('href') or item.get('url', ''),
            'snippet': item.get('body', '') or item.get('snippet', '')
        }
        for item in results_list
    ]


def _ddg_html_search(query: str, max_results: int) -> List[Dict[str, str]]:
    """Fallback: Search using DuckDuckGo HTML."""
    response = requests.get(
        'https://html.duckduckgo.com/html/',
        params={'q': query},
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()

    html = response.text
    results = []
    seen_urls = set()

    link_patterns = [
        r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)</a>',
        r'<a[^>]*class="[^"]*result[^"]*"[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
        r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
    ]

    for pattern in link_patterns:
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            if len(results) >= max_results:
                break

            url_match = match.group(1) if len(match.groups()) >= 1 else None
            title = match.group(2) if len(match.groups()) >= 2 else ''

            if not url_match or url_match in seen_urls:
                continue

            if any(skip in url_match.lower() for skip in ['duckduckgo.com', 'javascript:', 'data:', 'mailto:']):
                continue

            seen_urls.add(url_match)
            title = re.sub(r'&[^;]+;', '', title.strip()).replace('&nbsp;', ' ').strip()

            if title and len(title) > 3:
                results.append({'title': title, 'url': url_match, 'snippet': ''})

        if results:
            break

    return results


def _scrape_url(url: str, max_length: int = 10000) -> Dict[str, Any]:
    """Scrape content from a URL."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    html = response.text

    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else 'Untitled'

    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()

    for entity, char in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'),
                          ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")]:
        text = text.replace(entity, char)

    if len(text) > max_length:
        text = text[:max_length] + '...'

    return {'title': title, 'content': text}


@tool_wrapper(required_params=['query'])
def search_web_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the web using Google (Serper API) or DuckDuckGo.

    Priority: Serper API (Google) > DuckDuckGo library > HTML parsing fallback.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - max_results (int, optional): Max results (default: 10, max: 20)
            - provider (str, optional): 'serper' or 'duckduckgo' (default: auto)

    Returns:
        Dictionary with success, results, count, query, provider
    """
    query = params['query']
    max_results = min(params.get('max_results', 10), 20)
    provider_pref = params.get('provider', 'auto')

    # Priority 1: Serper API
    if SERPER_API_KEY and provider_pref in ('auto', 'serper'):
        try:
            results = _serper_search(query, max_results)
            if results:
                return tool_response(
                    results=results, count=len(results), query=query, provider='serper'
                )
        except Exception as e:
            logger.warning(f"Serper API failed: {e}, falling back to DuckDuckGo")

    # Priority 2: DuckDuckGo library
    if DDG_AVAILABLE and provider_pref in ('auto', 'duckduckgo'):
        try:
            results = _ddg_search(query, max_results)
            if results:
                return tool_response(
                    results=results, count=len(results), query=query, provider='duckduckgo'
                )
        except Exception as e:
            logger.warning(f"duckduckgo-search library failed: {e}, falling back to HTML")

    # Fallback: DuckDuckGo HTML
    try:
        results = _ddg_html_search(query, max_results)
        return tool_response(
            results=results, count=len(results), query=query, provider='duckduckgo_html'
        )
    except requests.RequestException as e:
        return tool_error(f'Network error: {str(e)}')


@tool_wrapper(required_params=['url'])
def fetch_webpage_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and extract text content from a web page.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to fetch
            - max_length (int, optional): Max content length (default: 10000)
            - max_retries (int, optional): Max retry attempts (default: 3)

    Returns:
        Dictionary with success, url, title, content, length
    """
    url = params['url']

    if '{' in url or '${' in url:
        return tool_error(f'URL contains unresolved template variables: {url}')

    max_length = params.get('max_length', 10000)
    max_retries = params.get('max_retries', 3)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            break
        except requests.Timeout:
            last_error = f'Request timed out (attempt {attempt + 1}/{max_retries})'
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return tool_error(f'Request timed out after {max_retries} attempts')
        except requests.RequestException as e:
            last_error = f'Network error: {str(e)}'
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return tool_error(f'Network error after {max_retries} attempts: {str(e)}')
    else:
        return tool_error(last_error or f'Failed after {max_retries} attempts')

    html = response.text

    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else 'Untitled'

    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()

    for entity, char in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'),
                          ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")]:
        text = text.replace(entity, char)

    if len(text) > max_length:
        text = text[:max_length] + '...'

    return tool_response(url=url, title=title, content=text, length=len(text))


@tool_wrapper(required_params=['query'])
def search_and_scrape_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search web and scrape content from top results.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - num_results (int, optional): Number of search results (default: 5)
            - scrape_top (int, optional): Number of results to scrape (default: 3)
            - max_content_length (int, optional): Max content per page (default: 5000)

    Returns:
        Dictionary with success, query, results, count, scraped_count, provider
    """
    query = params['query']
    num_results = params.get('num_results', 5)
    scrape_top = params.get('scrape_top', 3)
    max_content_length = params.get('max_content_length', 5000)

    search_result = search_web_tool({'query': query, 'max_results': num_results})

    if not search_result.get('success'):
        return search_result

    results = search_result.get('results', [])
    scraped_count = 0

    for result in results[:scrape_top]:
        url = result.get('url')
        if not url:
            continue

        try:
            scraped = _scrape_url(url, max_content_length)
            result['content'] = scraped.get('content', '')
            result['content_length'] = len(result['content'])
            scraped_count += 1
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            result['content'] = ''
            result['scrape_error'] = str(e)

    return tool_response(
        query=query,
        results=results,
        count=len(results),
        scraped_count=scraped_count,
        provider=search_result.get('provider', 'unknown')
    )


__all__ = ['search_web_tool', 'fetch_webpage_tool', 'search_and_scrape_tool']
