"""
Investing.com Commodities Price Fetcher Skill

Fetches latest commodities prices from investing.com.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


def get_commodities_prices_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch latest commodities prices from investing.com.
    
    Args:
        params: Dictionary containing:
            - category (str, optional): Category filter - 'energy', 'metals', 'agriculture', or 'all' (default: 'all')
            - format (str, optional): Output format - 'json', 'markdown', 'text' (default: 'markdown')
    
    Returns:
        Dictionary with:
            - success (bool): Whether fetch succeeded
            - commodities (list): List of commodities with prices
            - formatted_output (str): Formatted output string
            - timestamp (str): Fetch timestamp
            - error (str, optional): Error message if failed
    """
    try:
        category = params.get('category', 'all').lower()
        output_format = params.get('format', 'markdown')
        
        url = "https://www.investing.com/commodities/"
        
        # Headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        logger.info(f"Fetching commodities prices from {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        commodities = []
        
        # Find all commodity tables
        # Look for tables with commodity data
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this is a commodities table
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Try to find header row
            header_row = rows[0]
            headers_text = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Look for commodity-specific headers
            if any(keyword in ' '.join(headers_text).lower() for keyword in ['name', 'last', 'change', 'chg']):
                # This looks like a commodities table
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:
                        continue
                    
                    # Try to extract commodity name (usually in first cell, might be a link)
                    name_cell = cells[0]
                    name_link = name_cell.find('a')
                    if name_link:
                        name = name_link.get_text(strip=True)
                        commodity_url = name_link.get('href', '')
                    else:
                        name = name_cell.get_text(strip=True)
                        commodity_url = ''
                    
                    if not name or len(name) < 2:
                        continue
                    
                    # Extract price data
                    commodity_data = {
                        'name': name,
                        'url': commodity_url if commodity_url.startswith('http') else f"https://www.investing.com{commodity_url}" if commodity_url else '',
                    }
                    
                    # Try to extract last price, change, change %
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # Look for price-like values (numbers with decimals)
                    for i, text in enumerate(cell_texts[1:], start=1):
                        if not text:
                            continue
                        
                        # Try to identify what each cell contains
                        if i == 1 and any(char.isdigit() for char in text):
                            # Likely last price
                            commodity_data['last'] = text
                        elif '%' in text:
                            # Likely change %
                            commodity_data['change_pct'] = text
                        elif text.startswith(('+', '-')) and any(char.isdigit() for char in text):
                            # Likely change
                            commodity_data['change'] = text
                    
                    # Determine category
                    name_lower = name.lower()
                    if any(metal in name_lower for metal in ['gold', 'silver', 'copper', 'platinum', 'palladium']):
                        commodity_data['category'] = 'metals'
                    elif any(energy in name_lower for energy in ['oil', 'gas', 'heating', 'brent', 'crude', 'natural']):
                        commodity_data['category'] = 'energy'
                    elif any(agri in name_lower for agri in ['corn', 'wheat', 'coffee', 'sugar', 'cotton', 'cocoa', 'soybean']):
                        commodity_data['category'] = 'agriculture'
                    else:
                        commodity_data['category'] = 'other'
                    
                    commodities.append(commodity_data)
        
        # If we didn't find data in tables, try alternative approach
        if not commodities:
            # Look for specific commodity links/divs
            commodity_links = soup.find_all('a', href=lambda x: x and '/commodities/' in x)
            seen_names = set()
            
            for link in commodity_links:
                name = link.get_text(strip=True)
                if name and len(name) > 2 and name not in seen_names:
                    seen_names.add(name)
                    href = link.get('href', '')
                    
                    # Try to find nearby price data
                    parent = link.parent
                    price_text = ''
                    change_text = ''
                    
                    # Look in parent or siblings for price
                    if parent:
                        siblings = parent.find_next_siblings()
                        for sibling in siblings[:3]:
                            text = sibling.get_text(strip=True)
                            if any(char.isdigit() for char in text) and ('%' in text or '.' in text):
                                if '%' in text:
                                    change_text = text
                                else:
                                    price_text = text
                                break
                    
                    if name and (price_text or change_text):
                        commodities.append({
                            'name': name,
                            'url': f"https://www.investing.com{href}" if href and not href.startswith('http') else href,
                            'last': price_text if price_text else 'N/A',
                            'change_pct': change_text if change_text else 'N/A',
                            'category': 'unknown'
                        })
        
        # Filter by category if specified
        if category != 'all':
            commodities = [c for c in commodities if c.get('category') == category]
        
        # Remove duplicates
        seen = set()
        unique_commodities = []
        for c in commodities:
            name_key = c['name'].lower()
            if name_key not in seen:
                seen.add(name_key)
                unique_commodities.append(c)
        commodities = unique_commodities
        
        # Format output
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        if output_format == 'json':
            formatted_output = json.dumps(commodities, indent=2)
        elif output_format == 'markdown':
            formatted_output = f"# Commodities Prices\n\n"
            formatted_output += f"*Fetched: {timestamp}*\n\n"
            
            # Group by category
            by_category = {}
            for c in commodities:
                cat = c.get('category', 'other')
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(c)
            
            for cat in ['energy', 'metals', 'agriculture', 'other']:
                if cat in by_category:
                    formatted_output += f"## {cat.title()}\n\n"
                    formatted_output += "| Name | Last | Change | Change % |\n"
                    formatted_output += "|------|------|--------|----------|\n"
                    
                    for c in by_category[cat]:
                        name = c.get('name', 'N/A')
                        last = c.get('last', 'N/A')
                        change = c.get('change', 'N/A')
                        change_pct = c.get('change_pct', 'N/A')
                        formatted_output += f"| {name} | {last} | {change} | {change_pct} |\n"
                    formatted_output += "\n"
        else:  # text
            formatted_output = f"Commodities Prices (Fetched: {timestamp})\n\n"
            for c in commodities:
                formatted_output += f"{c.get('name', 'N/A')}: {c.get('last', 'N/A')} ({c.get('change_pct', 'N/A')})\n"
        
        return {
            'success': True,
            'commodities': commodities,
            'formatted_output': formatted_output,
            'timestamp': timestamp,
            'count': len(commodities)
        }
        
    except Exception as e:
        logger.error(f"Error fetching commodities prices: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to fetch commodities prices: {str(e)}',
            'commodities': [],
            'formatted_output': ''
        }


__all__ = ['get_commodities_prices_tool']
