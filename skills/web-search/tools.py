import os
import requests
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

# Serper API configuration (Google-quality results)
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

# Try to use duckduckgo-search library if available
DDG_AVAILABLE = False
try:
    # Try new package name first
    try:
        from ddgs import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        # Fallback to old package name
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False
    logger.info("duckduckgo-search/ddgs library not available, using HTML parsing fallback")


def serper_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search using Serper API (Google Search).

    Requires SERPER_API_KEY environment variable.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of results with title, url, snippet
    """
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY not set")

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        'q': query,
        'num': num_results
    }

    response = requests.post(
        'https://google.serper.dev/search',
        headers=headers,
        json=payload,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data.get('organic', []):
        results.append({
            'title': item.get('title', 'Untitled'),
            'url': item.get('link', ''),
            'snippet': item.get('snippet', '')
        })

    return results


def scrape_url(url: str, max_length: int = 10000) -> Dict[str, Any]:
    """
    Scrape content from a URL.

    Args:
        url: URL to scrape
        max_length: Maximum content length

    Returns:
        Dict with title and content
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    html = response.text

    # Extract title
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else 'Untitled'

    # Remove script/style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Extract text
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()

    # Decode HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")

    if len(text) > max_length:
        text = text[:max_length] + '...'

    return {'title': title, 'content': text}


def search_web_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the web using Google (Serper API) or DuckDuckGo.

    Priority: Serper API (Google) > DuckDuckGo library > HTML parsing fallback.
    Set SERPER_API_KEY for Google-quality results.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - max_results (int, optional): Maximum number of results (default: 10, max: 20)
            - provider (str, optional): 'serper' or 'duckduckgo' (default: auto)

    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - results (list): List of search results with title, url, snippet
            - count (int): Number of results
            - provider (str): Which provider was used
            - error (str, optional): Error message if failed
    """
    try:
        query = params.get('query')
        if not query:
            return {
                'success': False,
                'error': 'query parameter is required'
            }

        max_results = min(params.get('max_results', 10), 20)
        provider_pref = params.get('provider', 'auto')

        # Priority 1: Serper API (Google-quality results)
        if SERPER_API_KEY and provider_pref in ('auto', 'serper'):
            try:
                results = serper_search(query, max_results)
                if results:
                    return {
                        'success': True,
                        'results': results,
                        'count': len(results),
                        'query': query,
                        'provider': 'serper'
                    }
            except Exception as e:
                logger.warning(f"Serper API failed: {e}, falling back to DuckDuckGo")

        # Priority 2: Use duckduckgo-search library if available
        if DDG_AVAILABLE and provider_pref in ('auto', 'duckduckgo'):
            try:
                with DDGS() as ddgs:
                    results_list = list(ddgs.text(query, max_results=max_results))
                
                results = []
                for item in results_list:
                    # Handle both 'href' and 'url' keys
                    url = item.get('href') or item.get('url', '')
                    results.append({
                        'title': item.get('title', 'Untitled'),
                        'url': url,
                        'snippet': item.get('body', '') or item.get('snippet', '')
                    })
                
                if results:
                    return {
                        'success': True,
                        'results': results,
                        'count': len(results),
                        'query': query,
                        'provider': 'duckduckgo'
                    }
            except Exception as e:
                logger.warning(f"duckduckgo-search library failed: {e}, falling back to HTML")
                # Fall through to HTML parsing fallback
        
        # Fallback: Use DuckDuckGo HTML search (less reliable)
        url = 'https://html.duckduckgo.com/html/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params_data = {
            'q': query
        }
        
        response = requests.get(url, params=params_data, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML results - DuckDuckGo HTML structure may vary
        html = response.text
        
        results = []
        seen_urls = set()
        
        # Try to find result links - look for common patterns
        # Modern DuckDuckGo uses different structures
        link_patterns = [
            r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)</a>',
            r'<a[^>]*class="[^"]*result[^"]*"[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
        ]
        
        for pattern in link_patterns:
            matches = re.finditer(pattern, html, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if len(results) >= max_results:
                    break
                
                url_match = match.group(1) if len(match.groups()) >= 1 else None
                title = match.group(2) if len(match.groups()) >= 2 else ''
                
                if not url_match or url_match in seen_urls:
                    continue
                
                # Skip internal DuckDuckGo URLs and special URLs
                if any(skip in url_match.lower() for skip in ['duckduckgo.com', 'javascript:', 'data:', 'mailto:']):
                    continue
                
                seen_urls.add(url_match)
                
                # Clean up HTML entities
                title = re.sub(r'&[^;]+;', '', title.strip())
                title = title.replace('&nbsp;', ' ').strip()
                
                if title and len(title) > 3:  # Valid title
                    results.append({
                        'title': title,
                        'url': url_match,
                        'snippet': ''
                    })
            
            if results:
                break
        
        return {
            'success': True,
            'results': results,
            'count': len(results),
            'query': query,
            'provider': 'duckduckgo_html'
        }
    except requests.RequestException as e:
        return {
            'success': False,
            'error': f'Network error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error searching web: {str(e)}'
        }


def fetch_webpage_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and extract text content from a web page.
    
    Includes retry logic for network errors (3 attempts with exponential backoff).
    
    Args:
        params: Dictionary containing:
            - url (str, required): URL to fetch
            - max_length (int, optional): Maximum length of extracted text (default: 10000)
            - max_retries (int, optional): Maximum retry attempts (default: 3)
    
    Returns:
        Dictionary with:
            - success (bool): Whether fetch succeeded
            - url (str): Fetched URL
            - title (str): Page title
            - content (str): Extracted text content
            - length (int): Length of content
            - error (str, optional): Error message if failed
    """
    import time
    
    url = params.get('url')
    if not url:
        return {
            'success': False,
            'error': 'url parameter is required'
        }
    
    # Check if URL contains template variables (not resolved)
    if '{' in url or '${' in url:
        return {
            'success': False,
            'error': f'URL contains unresolved template variables: {url}. Ensure previous step outputs are properly referenced.'
        }
    
    max_length = params.get('max_length', 10000)
    max_retries = params.get('max_retries', 3)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Retry logic with exponential backoff
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            break  # Success, exit retry loop
        except requests.Timeout as e:
            last_error = f'Request timed out (attempt {attempt + 1}/{max_retries})'
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                return {
                    'success': False,
                    'error': f'Request timed out after {max_retries} attempts'
                }
        except requests.RequestException as e:
            last_error = f'Network error (attempt {attempt + 1}/{max_retries}): {str(e)}'
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return {
                    'success': False,
                    'error': f'Network error after {max_retries} attempts: {str(e)}'
                }
    else:
        # All retries exhausted
        return {
            'success': False,
            'error': last_error or f'Failed after {max_retries} attempts'
        }
    
    try:
        
        html = response.text
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else 'Untitled'
        
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract text from HTML (simple approach)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Decode HTML entities (basic)
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'content': text,
            'length': len(text)
        }
    except requests.Timeout:
        return {
            'success': False,
            'error': f'Request timed out after 15 seconds'
        }
    except requests.RequestException as e:
        return {
            'success': False,
            'error': f'Network error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error fetching webpage: {str(e)}'
        }


def search_and_scrape_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search web and scrape content from top results.

    Combines search with content scraping for deep research.
    Uses Serper API (Google) if SERPER_API_KEY is set.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - num_results (int, optional): Number of search results (default: 5)
            - scrape_top (int, optional): Number of results to scrape (default: 3)
            - max_content_length (int, optional): Max content per page (default: 5000)

    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - query (str): Search query
            - results (list): List of results with title, url, snippet, content
            - count (int): Number of results
            - scraped_count (int): Number of pages scraped
            - error (str, optional): Error message if failed
    """
    try:
        query = params.get('query')
        if not query:
            return {'success': False, 'error': 'query parameter is required'}

        num_results = params.get('num_results', 5)
        scrape_top = params.get('scrape_top', 3)
        max_content_length = params.get('max_content_length', 5000)

        # Search first
        search_result = search_web_tool({
            'query': query,
            'max_results': num_results
        })

        if not search_result.get('success'):
            return search_result

        results = search_result.get('results', [])
        scraped_count = 0

        # Scrape top results
        for i, result in enumerate(results[:scrape_top]):
            url = result.get('url')
            if not url:
                continue

            try:
                scraped = scrape_url(url, max_content_length)
                result['content'] = scraped.get('content', '')
                result['content_length'] = len(result['content'])
                scraped_count += 1
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
                result['content'] = ''
                result['scrape_error'] = str(e)

        return {
            'success': True,
            'query': query,
            'results': results,
            'count': len(results),
            'scraped_count': scraped_count,
            'provider': search_result.get('provider', 'unknown')
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Search and scrape failed: {str(e)}'
        }
