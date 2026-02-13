"""
Web Search Skill

Search the web using Google (Serper API) or DuckDuckGo.
Refactored to use Jotty core utilities.
"""

import os
import re
import time
import logging
import threading
import requests
from typing import Dict, Any, List, Optional

from Jotty.core.utils.env_loader import load_jotty_env, get_env
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

from Jotty.core.utils.skill_status import SkillStatus

load_jotty_env()

logger = logging.getLogger(__name__)

SERPER_API_KEY = get_env("SERPER_API_KEY")
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Try to load duckduckgo-search library

# Status emitter for progress updates
status = SkillStatus("web-search")

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


class SearchCache:
    """Thread-safe TTL cache for web search results.

    Prevents redundant API calls when the same query is executed
    multiple times within the TTL window (default 5 minutes).

    Usage::

        cache = SearchCache(ttl_seconds=300)
        cached = cache.get("serper:AI trends:10")
        if cached is None:
            results = _serper_search("AI trends", 10)
            cache.set("serper:AI trends:10", results)
    """

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}  # key → (result, timestamp)
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Return cached result if present and not expired, else None."""
        with self._lock:
            self._evict_expired()
            entry = self._cache.get(key)
            if entry is not None:
                return entry[0]
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a result with current timestamp."""
        with self._lock:
            self._cache[key] = (value, time.time())
            # Cap cache size to prevent unbounded growth
            if len(self._cache) > 200:
                self._evict_expired()
                if len(self._cache) > 200:
                    # Remove oldest entries
                    oldest = sorted(self._cache, key=lambda k: self._cache[k][1])
                    for k in oldest[:50]:
                        del self._cache[k]

    def _evict_expired(self) -> None:
        """Remove entries older than TTL. Must be called under lock."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()


_search_cache = SearchCache()


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
    """
    Scrape content from a URL with intelligent 403 handling.

    Uses SmartFetcher: direct → proxy retry on 403/429 → graceful skip.
    """
    from Jotty.core.utils.smart_fetcher import smart_fetch

    result = smart_fetch(url, timeout=15)

    if result.skipped or not result.success:
        raise requests.RequestException(result.error)

    html = result.content

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

    proxy_note = ' (via proxy)' if result.used_proxy else ''
    return {'title': title, 'content': text, 'fetched_via': 'proxy' if result.used_proxy else 'direct'}


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
    status.set_callback(params.pop('_status_callback', None))

    query = params['query']
    max_results = min(params.get('max_results', 10), 20)
    provider_pref = params.get('provider', 'auto')

    # Show search query (truncated)
    query_display = query[:50] + "..." if len(query) > 50 else query
    status.emit("Searching", f"🔍 {query_display}")

    # Check cache first
    cache_key = f"{provider_pref}:{query}:{max_results}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.info(f"Search cache HIT: {query_display}")
        return cached

    # Priority 1: Serper API
    if SERPER_API_KEY and provider_pref in ('auto', 'serper'):
        try:
            results = _serper_search(query, max_results)
            if results:
                response = tool_response(
                    results=results, count=len(results), query=query, provider='serper'
                )
                _search_cache.set(cache_key, response)
                return response
        except Exception as e:
            logger.warning(f"Serper API failed: {e}, falling back to DuckDuckGo")

    # Priority 2: DuckDuckGo library
    if DDG_AVAILABLE and provider_pref in ('auto', 'duckduckgo'):
        try:
            results = _ddg_search(query, max_results)
            if results:
                response = tool_response(
                    results=results, count=len(results), query=query, provider='duckduckgo'
                )
                _search_cache.set(cache_key, response)
                return response
        except Exception as e:
            logger.warning(f"duckduckgo-search library failed: {e}, falling back to HTML")

    # Fallback: DuckDuckGo HTML
    try:
        results = _ddg_html_search(query, max_results)
        response = tool_response(
            results=results, count=len(results), query=query, provider='duckduckgo_html'
        )
        _search_cache.set(cache_key, response)
        return response
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
    status.set_callback(params.pop('_status_callback', None))

    url = params['url']
    # Show URL being fetched (truncated)
    url_display = url[:60] + "..." if len(url) > 60 else url
    status.emit("Fetching", f"🌐 {url_display}")

    if '{' in url or '${' in url:
        return tool_error(f'URL contains unresolved template variables: {url}')

    max_length = params.get('max_length', 10000)

    # Smart fetch: direct → proxy on 403/429 → graceful skip
    from Jotty.core.utils.smart_fetcher import smart_fetch
    result = smart_fetch(url, timeout=15, max_proxy_attempts=2)

    if result.skipped:
        return tool_error(result.error)

    if not result.success:
        return tool_error(f'Failed to fetch: {result.error}')

    html = result.content

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

    return tool_response(
        url=url, title=title, content=text, length=len(text),
        fetched_via='proxy' if result.used_proxy else 'direct',
    )


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
    status.set_callback(params.pop('_status_callback', None))

    query = params['query']
    num_results = params.get('num_results', 5)
    scrape_top = params.get('scrape_top', 3)
    max_content_length = params.get('max_content_length', 5000)

    search_result = search_web_tool({'query': query, 'max_results': num_results})

    if not search_result.get('success'):
        return search_result

    results = search_result.get('results', [])
    scraped_count = 0

    # Parallelize URL scraping — biggest latency win (22s → ~5s for 3 URLs)
    to_scrape = [(i, r) for i, r in enumerate(results[:scrape_top]) if r.get('url')]
    if to_scrape:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scrape_one(idx_result):
            idx, result = idx_result
            url = result['url']
            try:
                scraped = _scrape_url(url, max_content_length)
                return idx, scraped.get('content', ''), None
            except Exception as e:
                return idx, '', str(e)

        with ThreadPoolExecutor(max_workers=min(len(to_scrape), 5)) as executor:
            futures = {executor.submit(_scrape_one, item): item for item in to_scrape}
            for future in as_completed(futures):
                idx, content, error = future.result()
                if content:
                    results[idx]['content'] = content
                    results[idx]['content_length'] = len(content)
                    scraped_count += 1
                elif error:
                    logger.warning(f"Failed to scrape {results[idx].get('url')}: {error}")
                    results[idx]['content'] = ''
                    results[idx]['scrape_error'] = error

    return tool_response(
        query=query,
        results=results,
        count=len(results),
        scraped_count=scraped_count,
        provider=search_result.get('provider', 'unknown')
    )


@tool_wrapper(required_params=['url'])
def serper_scrape_website_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrape a website using Serper API for clean markdown extraction.

    Unlike fetch_webpage_tool which does basic HTML scraping, this uses
    Serper's dedicated scraping endpoint for higher quality markdown output.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to scrape
            - max_length (int, optional): Max content length (default: 100000)

    Returns:
        Dictionary with success, url, title, content, content_length
    """
    status.set_callback(params.pop('_status_callback', None))

    url = params['url']
    max_length = params.get('max_length', 100000)

    if not SERPER_API_KEY:
        return tool_error('SERPER_API_KEY not set. Required for website scraping.')

    url_display = url[:60] + "..." if len(url) > 60 else url
    status.emit("Scraping", f"Scraping {url_display}")

    try:
        response = requests.post(
            'https://scrape.serper.dev/',
            headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
            json={'url': url},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        content = data.get('markdown', data.get('text', ''))
        title = data.get('title', '')

        if len(content) > max_length:
            content = content[:max_length] + '...'

        return tool_response(
            url=url,
            title=title,
            content=content,
            content_length=len(content),
            provider='serper_scrape'
        )
    except requests.RequestException as e:
        return tool_error(f'Scraping failed: {str(e)}')
    except Exception as e:
        return tool_error(f'Error scraping website: {str(e)}')


__all__ = ['search_web_tool', 'fetch_webpage_tool', 'search_and_scrape_tool', 'serper_scrape_website_tool', 'SearchCache']
