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

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)


# DRY: Use shared ProxyRotator from core (same logic, single source of truth)
from Jotty.core.utils.smart_fetcher import get_proxy_rotator, USER_AGENTS

# Status emitter for progress updates
status = SkillStatus("screener-financials")

# Use the shared global proxy rotator
_proxy_rotator = get_proxy_rotator()


def _get_random_user_agent() -> str:
    """Get random user agent. Uses shared USER_AGENTS list."""
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
                    _proxy_rotator.mark_failed(current_proxies.get('http', ''))
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
                    _proxy_rotator.mark_failed(current_proxies.get('http', ''))
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


@tool_wrapper()
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
    status.set_callback(params.pop('_status_callback', None))

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


@tool_wrapper()
def get_company_financials_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch financial data for a company from screener.in.
    
    Args:
        params: Dictionary containing:
            - company_name (str, required): Company name or screener.in code
            - data_type (str, optional): 'all', 'pl', 'balance_sheet', 'cash_flow', 'ratios',
                'shareholding', 'quarterly', 'peers', 'structured_ratios'
            - period (str, optional): 'annual' or 'quarterly', default: 'annual'
            - format (str, optional): 'json', 'markdown', 'csv', default: 'json'
            - use_proxy (bool, optional): Use proxy rotation, default: True
            - max_retries (int, optional): Maximum retries, default: 3
    
    Returns:
        Dictionary with financial data
    """
    status.set_callback(params.pop('_status_callback', None))

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
        
        # Extract Structured Ratios (parsed floats with normalized keys)
        if data_type in ['all', 'structured_ratios']:
            structured = _extract_structured_ratios(soup)
            financial_data['data']['structured_ratios'] = structured
        
        # Extract Shareholding Pattern
        if data_type in ['all', 'shareholding']:
            shareholding = _extract_shareholding(soup)
            financial_data['data']['shareholding'] = shareholding
        
        # Extract Quarterly Results
        if data_type in ['all', 'quarterly']:
            quarterly = _extract_quarterly_results(soup)
            financial_data['data']['quarterly_results'] = quarterly
        
        # Extract Peer Companies
        if data_type in ['all', 'peers']:
            peers = _extract_peers(soup)
            financial_data['data']['peers'] = peers
        
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


def _parse_float_value(value: str) -> Optional[float]:
    """Safely parse a float value, handling Cr/L suffixes."""
    if not value:
        return None
    try:
        value = value.strip().replace(',', '').replace('%', '').replace('₹', '')
        multiplier = 1
        if value.endswith('Cr'):
            value = value[:-2]
            multiplier = 10000000  # 1 crore
        elif value.endswith('L'):
            value = value[:-1]
            multiplier = 100000  # 1 lakh
        return float(value) * multiplier
    except (ValueError, TypeError):
        return None


def _extract_shareholding(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract shareholding pattern from screener.in HTML."""
    shareholding = {}
    try:
        sh_section = soup.find('section', id='shareholding')
        if not sh_section:
            return shareholding

        table = sh_section.find('table')
        if not table:
            return shareholding

        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                name = cells[0].get_text(strip=True).lower()
                # Get the latest value (last non-empty column)
                values = [c.get_text(strip=True) for c in cells[1:]]
                latest_value = None
                for v in reversed(values):
                    if v and v != '-':
                        latest_value = v.replace('%', '').strip()
                        break

                if latest_value:
                    if 'promoter' in name:
                        shareholding['promoter'] = _parse_float_value(latest_value)
                    elif 'fii' in name or 'foreign' in name:
                        shareholding['fii'] = _parse_float_value(latest_value)
                    elif 'dii' in name or 'domestic' in name:
                        shareholding['dii'] = _parse_float_value(latest_value)
                    elif 'public' in name:
                        shareholding['public'] = _parse_float_value(latest_value)
    except Exception as e:
        logger.debug(f"Shareholding extraction error: {e}")
    return shareholding


def _extract_quarterly_results(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract quarterly results from screener.in HTML."""
    results = []
    try:
        qr_section = soup.find('section', id='quarters')
        if not qr_section:
            return results

        table = qr_section.find('table')
        if not table:
            return results

        # Get headers (quarter names)
        headers = []
        header_row = table.find('thead')
        if header_row:
            ths = header_row.find_all('th')
            headers = [th.get_text(strip=True) for th in ths]

        tbody = table.find('tbody')
        if not tbody:
            return results

        rows = tbody.find_all('tr')
        for row in rows[:5]:  # Key metrics only
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                metric = cells[0].get_text(strip=True)
                values = [_parse_float_value(c.get_text(strip=True).replace(',', ''))
                          for c in cells[1:]]

                if metric.lower() in ['sales', 'revenue', 'net profit', 'operating profit', 'eps']:
                    for i, val in enumerate(values[:4]):
                        if len(results) <= i:
                            results.append({'quarter': headers[i + 1] if i + 1 < len(headers) else f'Q{i + 1}'})
                        results[i][metric.lower().replace(' ', '_')] = val

    except Exception as e:
        logger.debug(f"Quarterly results extraction error: {e}")
    return results[:4]


def _extract_peers(soup: BeautifulSoup) -> List[str]:
    """Extract peer company tickers from screener.in HTML."""
    peers = []
    try:
        peer_section = soup.find('section', id='peers')
        if not peer_section:
            return peers

        links = peer_section.find_all('a', href=True)
        for link in links:
            href = link.get('href', '')
            if '/company/' in href:
                parts = href.strip('/').split('/')
                if len(parts) >= 2:
                    peer_ticker = parts[-1].upper()
                    if peer_ticker and peer_ticker not in peers:
                        peers.append(peer_ticker)
    except Exception as e:
        logger.debug(f"Peers extraction error: {e}")
    return peers[:10]


def _extract_structured_ratios(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract financial ratios as parsed floats with normalized keys.

    Returns dict like: {pe_ratio: 25.3, pb_ratio: 3.5, roce: 18.2, ...}
    """
    ratios: Dict[str, Any] = {}
    try:
        # Pattern 1: ul.flex-list items
        ratio_lists = soup.find_all('ul', class_='flex-list')
        for ul in ratio_lists:
            items = ul.find_all('li')
            for item in items:
                name_elem = item.find('span', class_='name')
                value_elem = item.find('span', class_='number')
                if name_elem and value_elem:
                    name = name_elem.get_text(strip=True).lower()
                    value = value_elem.get_text(strip=True)
                    value = value.replace(',', '').replace('%', '').replace('₹', '').strip()

                    if 'p/e' in name or 'pe' in name:
                        ratios['pe_ratio'] = _parse_float_value(value)
                    elif 'p/b' in name or 'pb' in name:
                        ratios['pb_ratio'] = _parse_float_value(value)
                    elif 'roce' in name:
                        ratios['roce'] = _parse_float_value(value)
                    elif 'roe' in name:
                        ratios['roe'] = _parse_float_value(value)
                    elif 'debt' in name and 'equity' in name:
                        ratios['debt_equity'] = _parse_float_value(value)
                    elif 'market cap' in name:
                        ratios['market_cap'] = _parse_float_value(value)
                    elif 'current price' in name or 'stock' in name:
                        ratios['current_price'] = _parse_float_value(value)
                    elif 'book value' in name:
                        ratios['book_value'] = _parse_float_value(value)
                    elif 'dividend yield' in name:
                        ratios['dividend_yield'] = _parse_float_value(value)

        # Pattern 2: top-ratios section (fallback)
        top_ratios = soup.find('div', id='top-ratios')
        if top_ratios:
            spans = top_ratios.find_all('span')
            for i in range(0, len(spans) - 1, 2):
                name = spans[i].get_text(strip=True).lower()
                value = spans[i + 1].get_text(strip=True)
                value = value.replace(',', '').replace('%', '').replace('₹', '').strip()

                if 'pe' in name and 'pe_ratio' not in ratios:
                    ratios['pe_ratio'] = _parse_float_value(value)
                elif 'roce' in name and 'roce' not in ratios:
                    ratios['roce'] = _parse_float_value(value)
    except Exception as e:
        logger.debug(f"Structured ratio extraction error: {e}")
    return ratios


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


@tool_wrapper()
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
    status.set_callback(params.pop('_status_callback', None))

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
