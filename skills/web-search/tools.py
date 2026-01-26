import requests
from typing import Dict, Any
from urllib.parse import urljoin, urlparse
import re


def search_web_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo (no API key required).
    
    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - max_results (int, optional): Maximum number of results (default: 10, max: 20)
    
    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - results (list): List of search results with title, url, snippet
            - count (int): Number of results
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
        
        # Use DuckDuckGo HTML search (no API key needed)
        url = 'https://html.duckduckgo.com/html/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        params_data = {
            'q': query
        }
        
        response = requests.get(url, params=params_data, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML results (simple regex-based extraction)
        html = response.text
        
        results = []
        
        # DuckDuckGo HTML structure: results are in <div class="result">
        # Extract title, link, and snippet
        result_pattern = r'<div class="result[^"]*">.*?<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
        
        matches = re.finditer(result_pattern, html, re.DOTALL)
        
        for match in matches:
            if len(results) >= max_results:
                break
            
            url_match = match.group(1)
            title = match.group(2).strip()
            snippet = match.group(3).strip()
            
            # Clean up HTML entities
            title = re.sub(r'&[^;]+;', '', title)
            snippet = re.sub(r'&[^;]+;', '', snippet)
            
            results.append({
                'title': title,
                'url': url_match,
                'snippet': snippet
            })
        
        # Fallback: if regex didn't work, try simpler extraction
        if not results:
            # Try alternative pattern
            link_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            links = re.findall(link_pattern, html)
            
            for url_match, title in links[:max_results]:
                title = re.sub(r'&[^;]+;', '', title.strip())
                results.append({
                    'title': title,
                    'url': url_match,
                    'snippet': ''
                })
        
        return {
            'success': True,
            'results': results,
            'count': len(results),
            'query': query
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
    
    Args:
        params: Dictionary containing:
            - url (str, required): URL to fetch
            - max_length (int, optional): Maximum length of extracted text (default: 10000)
    
    Returns:
        Dictionary with:
            - success (bool): Whether fetch succeeded
            - url (str): Fetched URL
            - title (str): Page title
            - content (str): Extracted text content
            - length (int): Length of content
            - error (str, optional): Error message if failed
    """
    try:
        url = params.get('url')
        if not url:
            return {
                'success': False,
                'error': 'url parameter is required'
            }
        
        max_length = params.get('max_length', 10000)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
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
