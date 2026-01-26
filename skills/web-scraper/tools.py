import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List
import html2text
import re
from urllib.parse import urljoin, urlparse


def scrape_website_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrape a website and extract content.
    
    Args:
        params: Dictionary containing:
            - url (str, required): URL to scrape
            - follow_links (bool, optional): Follow links (spider mode), default: False
            - max_pages (int, optional): Maximum pages in spider mode, default: 10
            - output_format (str, optional): Output format, default: 'markdown'
            - selectors (dict, optional): Custom CSS selectors
            - exclude_patterns (list, optional): URL patterns to exclude
            - proxy (str, optional): Proxy URL
            - timeout (int, optional): Request timeout, default: 30
    
    Returns:
        Dictionary with:
            - success (bool): Whether scraping succeeded
            - url (str): Scraped URL
            - title (str): Page title
            - content (str): Extracted content
            - pages_scraped (int): Number of pages scraped
            - error (str, optional): Error message if failed
    """
    try:
        url = params.get('url')
        if not url:
            return {
                'success': False,
                'error': 'url parameter is required'
            }
        
        follow_links = params.get('follow_links', False)
        max_pages = params.get('max_pages', 10)
        output_format = params.get('output_format', 'markdown')
        selectors = params.get('selectors', {})
        exclude_patterns = params.get('exclude_patterns', [])
        proxy = params.get('proxy')
        timeout = params.get('timeout', 30)
        
        # Headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Proxy configuration
        proxies = None
        if proxy:
            proxies = {
                'http': proxy,
                'https': proxy
            }
        
        if follow_links:
            # Spider mode - scrape multiple pages
            return _scrape_spider_mode(
                url, max_pages, output_format, selectors,
                exclude_patterns, headers, proxies, timeout
            )
        else:
            # Single page mode
            return _scrape_single_page(
                url, output_format, selectors, headers, proxies, timeout
            )
    except Exception as e:
        return {
            'success': False,
            'error': f'Error scraping website: {str(e)}'
        }


def _scrape_single_page(
    url: str,
    output_format: str,
    selectors: Dict[str, str],
    headers: Dict[str, str],
    proxies: Optional[Dict[str, str]],
    timeout: int
) -> Dict[str, Any]:
    """Scrape a single page"""
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
            # Default: extract main content
            main = soup.find('main') or soup.find('article') or soup.find('body')
            if main:
                # Remove script and style tags
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
        # 'text' format is already extracted above
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'content': content[:100000],  # Limit content size
            'content_length': len(content),
            'pages_scraped': 1
        }
    except requests.RequestException as e:
        return {
            'success': False,
            'error': f'Network error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error scraping page: {str(e)}'
        }


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
    """Scrape multiple pages following links"""
    try:
        visited = set()
        to_visit = [start_url]
        pages_data = []
        domain = urlparse(start_url).netloc
        
        while to_visit and len(pages_data) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            # Check exclude patterns
            if any(pattern in url for pattern in exclude_patterns):
                continue
            
            visited.add(url)
            
            # Scrape page
            page_result = _scrape_single_page(
                url, output_format, selectors, headers, proxies, timeout
            )
            
            if page_result.get('success'):
                pages_data.append(page_result)
                
                # Extract links for next pages (if not at max)
                if len(pages_data) < max_pages:
                    try:
                        response = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(url, href)
                            
                            # Only follow same-domain links
                            if urlparse(full_url).netloc == domain:
                                if full_url not in visited and full_url not in to_visit:
                                    to_visit.append(full_url)
                    except:
                        pass  # Continue even if link extraction fails
        
        # Combine all pages
        combined_content = '\n\n---\n\n'.join([
            f"# {p.get('title', 'Untitled')}\n\n{p.get('content', '')}"
            for p in pages_data
        ])
        
        return {
            'success': True,
            'url': start_url,
            'title': f'Scraped {len(pages_data)} pages',
            'content': combined_content[:200000],  # Limit size
            'content_length': len(combined_content),
            'pages_scraped': len(pages_data),
            'pages': [{'url': p['url'], 'title': p['title']} for p in pages_data]
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error in spider mode: {str(e)}'
        }
