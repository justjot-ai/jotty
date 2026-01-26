"""
Screener.in Financials Skill

Fetches financial data for Indian companies from screener.in.
Includes free proxy rotation to avoid blocking.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List
import time
import random
import re
from urllib.parse import quote, urljoin
import json
import logging

logger = logging.getLogger(__name__)


# Free proxy sources (rotated automatically)
FREE_PROXY_SOURCES = [
    # Free proxy list APIs
    "https://api.proxyscrape.com/v2/?request=get&protocol=http",
    "https://www.proxy-list.download/api/v1/get?type=http",
]

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
]


class ProxyRotator:
    """Manages free proxy rotation for screener.in requests."""
    
    def __init__(self):
        self.proxies: List[str] = []
        self.current_proxy_index = 0
        self.failed_proxies: set = set()
        self.last_fetch_time = 0
        self.proxy_cache_duration = 3600  # 1 hour
    
    def _fetch_free_proxies(self) -> List[str]:
        """Fetch free proxies from public sources."""
        proxies = []
        
        # Try multiple free proxy sources
        proxy_sources = [
            # Geonode API (free tier) - Most reliable
            ("https://proxylist.geonode.com/api/proxy-list?limit=20&page=1&sort_by=lastChecked&sort_type=desc&protocols=http", "geonode"),
            # ProxyScrape API (alternative format)
            ("https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all", "proxyscrape"),
            # FreeProxyList API
            ("https://www.proxy-list.download/api/v1/get?type=http", "proxy-list"),
        ]
        
        for url, source_name in proxy_sources:
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    content = response.text.strip()
                    
                    # Parse different formats
                    if source_name == "geonode":
                        # JSON format
                        try:
                            data = response.json()
                            for proxy in data.get('data', [])[:15]:
                                ip = proxy.get('ip')
                                port = proxy.get('port')
                                if ip and port:
                                    proxies.append(f"http://{ip}:{port}")
                        except Exception as e:
                            logger.debug(f"Failed to parse Geonode JSON: {e}")
                    else:
                        # Text format (IP:PORT per line)
                        proxy_list = content.split('\n')
                        for proxy in proxy_list[:15]:
                            proxy = proxy.strip()
                            # Skip invalid responses
                            if 'invalid' in proxy.lower() or 'error' in proxy.lower():
                                continue
                            if proxy and ':' in proxy and not proxy.startswith('http'):
                                proxies.append(f"http://{proxy}")
                            elif proxy and proxy.startswith('http'):
                                proxies.append(proxy)
                    
                    if proxies:
                        logger.info(f"Fetched {len(proxies)} proxies from {source_name}")
                        break
            except Exception as e:
                logger.debug(f"Failed to fetch proxies from {source_name}: {e}")
                continue
        
        # If no proxies found, return empty list (will use direct connection)
        if not proxies:
            logger.info("No free proxies available, using direct connection")
        
        return proxies[:20]  # Limit to 20 proxies
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy in rotation."""
        current_time = time.time()
        
        # Refresh proxy list if cache expired
        if not self.proxies or (current_time - self.last_fetch_time) > self.proxy_cache_duration:
            self.proxies = self._fetch_free_proxies()
            self.last_fetch_time = current_time
            self.failed_proxies.clear()
        
        if not self.proxies:
            return None
        
        # Try next proxy
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_proxy_index % len(self.proxies)]
            self.current_proxy_index += 1
            
            if proxy not in self.failed_proxies:
                return {
                    'http': proxy,
                    'https': proxy
                }
            
            attempts += 1
        
        # All proxies failed, reset and try again
        self.failed_proxies.clear()
        if self.proxies:
            proxy = self.proxies[0]
            return {
                'http': proxy,
                'https': proxy
            }
        
        return None
    
    def mark_proxy_failed(self, proxy: str):
        """Mark a proxy as failed."""
        self.failed_proxies.add(proxy)


# Global proxy rotator instance
_proxy_rotator = ProxyRotator()


def _get_random_user_agent() -> str:
    """Get random user agent."""
    return random.choice(USER_AGENTS)


def _make_request(url: str, use_proxy: bool = True, max_retries: int = 3) -> Optional[requests.Response]:
    """Make HTTP request with proxy rotation and retry logic."""
    headers = {
        'User-Agent': _get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    proxies = None
    if use_proxy:
        proxies = _proxy_rotator.get_proxy()
        if proxies:
            logger.debug(f"Using proxy: {proxies.get('http', 'none')}")
    
    # Try with proxy first, then fallback to direct connection
    proxy_attempts = 0
    max_proxy_attempts = 2 if use_proxy else 0
    
    for attempt in range(max_retries):
        try:
            # Try proxy first (if enabled and available)
            current_proxies = proxies if (use_proxy and proxy_attempts < max_proxy_attempts) else None
            
            response = requests.get(
                url,
                headers=headers,
                proxies=current_proxies,
                timeout=15,
                allow_redirects=True
            )
            
            # Check if blocked
            if response.status_code == 403 or 'blocked' in response.text.lower():
                if current_proxies:
                    _proxy_rotator.mark_proxy_failed(current_proxies.get('http', ''))
                    proxy_attempts += 1
                    if proxy_attempts < max_proxy_attempts:
                        proxies = _proxy_rotator.get_proxy()
                        time.sleep(1)
                        continue
                    else:
                        # Fallback to direct connection
                        logger.info("Proxies failed, falling back to direct connection")
                        proxies = None
                        continue
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            # If proxy error, try direct connection
            if current_proxies and ('proxy' in str(e).lower() or 'tunnel' in str(e).lower()):
                if current_proxies:
                    _proxy_rotator.mark_proxy_failed(current_proxies.get('http', ''))
                proxy_attempts += 1
                
                if proxy_attempts < max_proxy_attempts:
                    proxies = _proxy_rotator.get_proxy()
                    logger.debug(f"Proxy failed, trying next proxy: {e}")
                    time.sleep(1)
                    continue
                else:
                    # Fallback to direct connection
                    logger.info("All proxies failed, using direct connection")
                    proxies = None
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logger.debug(f"Request failed, retrying in {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed after {max_retries} attempts: {e}")
                return None
    
    return None


def search_company_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for a company on screener.in.
    
    Args:
        params: Dictionary containing:
            - query (str, required): Company name or search query
            - max_results (int, optional): Maximum results, default: 10
    
    Returns:
        Dictionary with search results
    """
    try:
        query = params.get('query')
        if not query:
            return {
                'success': False,
                'error': 'query parameter is required'
            }
        
        max_results = params.get('max_results', 10)
        
        # Screener.in search URL
        search_url = f"https://www.screener.in/api/company/search/?q={quote(query)}"
        
        logger.info(f"Searching screener.in for: {query}")
        
        response = _make_request(search_url, use_proxy=True)
        if not response:
            return {
                'success': False,
                'error': 'Failed to fetch search results'
            }
        
        try:
            data = response.json()
            results = []
            
            # Handle both list and dict responses
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('results', [])
            else:
                items = []
            
            for item in items[:max_results]:
                # Handle both dict and list item formats
                if isinstance(item, dict):
                    # Extract company code from URL if available
                    url = item.get('url', '')
                    if url:
                        # Extract code from URL like "/company/RELIANCE/consolidated/"
                        url_parts = url.strip('/').split('/')
                        if 'company' in url_parts:
                            company_idx = url_parts.index('company')
                            if company_idx + 1 < len(url_parts):
                                item_code = url_parts[company_idx + 1]
                            else:
                                item_code = str(item.get('id', ''))
                        else:
                            item_code = str(item.get('id', ''))
                    else:
                        item_code = str(item.get('id', ''))
                    
                    item_name = item.get('name') or item.get('company_name', '')
                else:
                    # If item is a string or other format
                    item_code = str(item) if item else ''
                    item_name = str(item) if item else ''
                
                if item_code:
                    results.append({
                        'name': item_name or item_code,
                        'code': item_code,
                        'url': f"https://www.screener.in/company/{item_code}/",
                        'industry': item.get('industry', '') if isinstance(item, dict) else '',
                    })
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'count': len(results)
            }
        except json.JSONDecodeError:
            # Fallback: parse HTML if API doesn't work
            return {
                'success': False,
                'error': 'Failed to parse search results',
                'message': 'Screener.in API may have changed or blocked the request'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Error searching company: {str(e)}'
        }


def get_company_financials_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch financial data for a company from screener.in.
    
    Args:
        params: Dictionary containing:
            - company_name (str, required): Company name or screener.in code
            - data_type (str, optional): 'all', 'pl', 'balance_sheet', 'cash_flow', 'ratios'
            - period (str, optional): 'annual' or 'quarterly', default: 'annual'
            - format (str, optional): 'json', 'markdown', 'csv', default: 'json'
            - use_proxy (bool, optional): Use proxy rotation, default: True
            - max_retries (int, optional): Maximum retries, default: 3
    
    Returns:
        Dictionary with financial data
    """
    try:
        company_name = params.get('company_name')
        if not company_name:
            return {
                'success': False,
                'error': 'company_name parameter is required'
            }
        
        data_type = params.get('data_type', 'all')
        period = params.get('period', 'annual')
        output_format = params.get('format', 'json')
        use_proxy = params.get('use_proxy', True)
        max_retries = params.get('max_retries', 3)
        
        # Get company code if needed
        company_code = str(company_name)
        if not company_code.replace('/', '').replace('-', '').isalnum() and '/' not in company_code:
            # Search for company first
            search_result = search_company_tool({'query': company_name, 'max_results': 1})
            if search_result.get('success') and search_result.get('results'):
                company_code = search_result['results'][0]['code']
                company_name = search_result['results'][0]['name']
            else:
                return {
                    'success': False,
                    'error': f'Company not found: {company_name}'
                }
        
        # Build URL
        base_url = f"https://www.screener.in/company/{company_code}/"
        if period == 'quarterly':
            base_url += "quarterly/"
        
        logger.info(f"Fetching financials for {company_name} ({company_code})")
        
        # Fetch page
        response = _make_request(base_url, use_proxy=use_proxy, max_retries=max_retries)
        if not response:
            return {
                'success': False,
                'error': 'Failed to fetch company page'
            }
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract financial data
        financial_data = {
            'company_name': company_name,
            'company_code': company_code,
            'period': period,
            'data': {}
        }
        
        # Extract P&L data
        if data_type in ['all', 'pl']:
            pl_data = _extract_pl_data(soup)
            financial_data['data']['profit_loss'] = pl_data
        
        # Extract Balance Sheet data
        if data_type in ['all', 'balance_sheet']:
            bs_data = _extract_balance_sheet_data(soup)
            financial_data['data']['balance_sheet'] = bs_data
        
        # Extract Cash Flow data
        if data_type in ['all', 'cash_flow']:
            cf_data = _extract_cash_flow_data(soup)
            financial_data['data']['cash_flow'] = cf_data
        
        # Extract Ratios
        if data_type in ['all', 'ratios']:
            ratios = _extract_ratios(soup)
            financial_data['data']['ratios'] = ratios
        
        # Convert to requested format
        if output_format == 'markdown':
            financial_data['formatted'] = _format_as_markdown(financial_data)
        elif output_format == 'csv':
            financial_data['formatted'] = _format_as_csv(financial_data)
        
        return {
            'success': True,
            **financial_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching financials: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Error fetching financials: {str(e)}'
        }


def _extract_pl_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract Profit & Loss data from soup."""
    pl_data = {}
    
    # Screener.in uses tables with class "data-table responsive-text-nowrap"
    # P&L tables typically contain "Sales", "Expenses", "Operating Profit", etc.
    tables = soup.find_all('table', class_='data-table')
    
    for table in tables:
        # Check if this is a P&L table by looking for P&L keywords
        table_text = table.get_text().lower()
        pl_keywords = ['sales', 'revenue', 'expenses', 'operating profit', 'net profit', 'ebitda']
        
        if any(keyword in table_text for keyword in pl_keywords):
            headers = []
            rows = []
            
            # Extract headers from thead
            thead = table.find('thead')
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all(['th', 'td'])]
            
            # Extract data rows (skip header row if no thead)
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells and len(cells) > 1:  # Skip empty rows
                    rows.append(cells)
            
            if headers and rows:
                pl_data['headers'] = headers
                pl_data['rows'] = rows
                break  # Found P&L table
    
    return pl_data


def _extract_balance_sheet_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract Balance Sheet data from soup."""
    bs_data = {}
    
    # Balance Sheet tables contain keywords like "Equity", "Assets", "Liabilities", "Borrowings"
    tables = soup.find_all('table', class_='data-table')
    
    for table in tables:
        table_text = table.get_text().lower()
        bs_keywords = ['equity capital', 'reserves', 'borrowings', 'assets', 'liabilities', 'share capital']
        
        if any(keyword in table_text for keyword in bs_keywords):
            headers = []
            rows = []
            
            thead = table.find('thead')
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all(['th', 'td'])]
            
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells and len(cells) > 1:
                    rows.append(cells)
            
            if headers and rows:
                bs_data['headers'] = headers
                bs_data['rows'] = rows
                break  # Found Balance Sheet table
    
    return bs_data


def _extract_cash_flow_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract Cash Flow data from soup."""
    cf_data = {}
    
    # Cash Flow tables contain keywords like "Operating", "Investing", "Financing", "Cash"
    tables = soup.find_all('table', class_='data-table')
    
    for table in tables:
        table_text = table.get_text().lower()
        cf_keywords = ['operating activities', 'investing activities', 'financing activities', 'cash flow', 'net cash']
        
        if any(keyword in table_text for keyword in cf_keywords):
            headers = []
            rows = []
            
            thead = table.find('thead')
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all(['th', 'td'])]
            
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells and len(cells) > 1:
                    rows.append(cells)
            
            if headers and rows:
                cf_data['headers'] = headers
                cf_data['rows'] = rows
                break  # Found Cash Flow table
    
    return cf_data


def _extract_ratios(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract key financial ratios from screener.in HTML."""
    ratios = {}
    
    # Pattern 1: Ratios in <li class="flex flex-space-between"> with <span class="name"> and <span class="number">
    ratio_items = soup.find_all('li', class_=lambda x: x and 'flex' in str(x) and 'flex-space-between' in str(x))
    
    for item in ratio_items:
        name_span = item.find('span', class_='name')
        number_span = item.find('span', class_='number')
        
        if name_span and number_span:
            ratio_name = name_span.get_text(strip=True)
            ratio_value = number_span.get_text(strip=True)
            
            # Filter for actual ratios (not navigation items)
            if ratio_name and ratio_value and len(ratio_name) < 50:  # Reasonable ratio name length
                ratios[ratio_name] = ratio_value
    
    # Pattern 2: Growth metrics in ranges-table
    ranges_tables = soup.find_all('table', class_='ranges-table')
    for table in ranges_tables:
        rows = table.find_all('tr')
        metric_name = None
        
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if len(cells) >= 2:
                # First row usually has metric name
                if not metric_name and cells[0] and ':' not in cells[0]:
                    metric_name = cells[0]
                elif cells[0] and ':' in cells[0]:
                    # Format: "10 Years: 5%"
                    period = cells[0].replace(':', '').strip()
                    value = cells[1].strip()
                    if metric_name and period and value:
                        ratios[f"{metric_name} ({period})"] = value
                elif metric_name and cells[0] and cells[1]:
                    # Format: period in first cell, value in second
                    ratios[f"{metric_name} ({cells[0]})"] = cells[1]
    
    return ratios


def _format_as_markdown(data: Dict[str, Any]) -> str:
    """Format financial data as markdown."""
    lines = [f"# {data['company_name']} - Financial Data"]
    lines.append(f"**Period:** {data['period']}")
    lines.append("")
    
    for section_name, section_data in data['data'].items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")
        
        if isinstance(section_data, dict):
            if 'headers' in section_data and 'rows' in section_data:
                # Table format
                headers = section_data['headers']
                rows = section_data['rows']
                
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                
                for row in rows:
                    lines.append("| " + " | ".join(row[:len(headers)]) + " |")
            else:
                # Key-value format
                for key, value in section_data.items():
                    lines.append(f"- **{key}:** {value}")
        
        lines.append("")
    
    return "\n".join(lines)


def _format_as_csv(data: Dict[str, Any]) -> str:
    """Format financial data as CSV."""
    lines = []
    
    for section_name, section_data in data['data'].items():
        lines.append(f"# {section_name}")
        
        if isinstance(section_data, dict) and 'headers' in section_data:
            headers = section_data['headers']
            rows = section_data['rows']
            
            lines.append(",".join(headers))
            for row in rows:
                lines.append(",".join([str(cell) for cell in row[:len(headers)]]))
        
        lines.append("")
    
    return "\n".join(lines)


def get_company_ratios_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch key financial ratios for a company.
    
    Args:
        params: Dictionary containing:
            - company_name (str, required): Company name or code
            - period (str, optional): 'annual' or 'quarterly', default: 'annual'
    
    Returns:
        Dictionary with ratios
    """
    try:
        company_name = params.get('company_name')
        if not company_name:
            return {
                'success': False,
                'error': 'company_name parameter is required'
            }
        
        # Use get_company_financials_tool with ratios only
        result = get_company_financials_tool({
            'company_name': company_name,
            'data_type': 'ratios',
            'period': params.get('period', 'annual'),
            'format': 'json'
        })
        
        if result.get('success'):
            return {
                'success': True,
                'company_name': result.get('company_name'),
                'company_code': result.get('company_code'),
                'ratios': result.get('data', {}).get('ratios', {})
            }
        else:
            return result
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Error fetching ratios: {str(e)}'
        }
