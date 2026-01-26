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
        
        # Find all links to commodities pages
        commodity_links = soup.find_all('a', href=lambda x: x and '/commodities/' in x.lower())
        
        commodity_keywords = {
            'metals': ['gold', 'silver', 'copper', 'platinum', 'palladium'],
            'energy': ['oil', 'brent', 'crude', 'gas', 'natural gas', 'heating oil', 'heating'],
            'agriculture': ['corn', 'wheat', 'coffee', 'sugar', 'cotton', 'cocoa', 'soybean', 'soy']
        }
        
        seen_commodities = set()
        
        # Extract from commodity links
        for link in commodity_links:
            name = link.get_text(strip=True)
            href = link.get('href', '')
            
            # Skip if empty or already seen
            if not name or len(name) < 2 or name.lower() in seen_commodities:
                continue
            
            # Skip stock tickers (short all-caps)
            if len(name) <= 5 and name.isupper() and not any(kw in name.lower() for kw_list in commodity_keywords.values() for kw in kw_list):
                continue
            
            # Skip if not a commodity name
            name_lower = name.lower()
            is_commodity = any(
                kw in name_lower for kw_list in commodity_keywords.values() for kw in kw_list
            ) or '/commodities/' in href.lower()
            
            if not is_commodity:
                continue
            
            seen_commodities.add(name.lower())
            
            # Determine category
            category = 'other'
            if any(kw in name_lower for kw in commodity_keywords['metals']):
                category = 'metals'
            elif any(kw in name_lower for kw in commodity_keywords['energy']):
                category = 'energy'
            elif any(kw in name_lower for kw in commodity_keywords['agriculture']):
                category = 'agriculture'
            
            # Try to find price data near the link
            # Look in parent row or nearby elements
            parent = link.parent
            price_data = {}
            
            # Check if parent is a table row
            if parent and parent.name == 'td':
                row = parent.parent if parent.parent and parent.parent.name == 'tr' else None
                if row:
                    cells = row.find_all(['td', 'th'])
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # Find price, change, change% in cells
                    for text in cell_texts:
                        if not text:
                            continue
                        
                        # Last price: number with decimal, no %
                        if '.' in text and any(char.isdigit() for char in text) and '%' not in text and '+' not in text and '-' not in text:
                            try:
                                # Try to parse as float
                                float(text.replace(',', ''))
                                if 'last' not in price_data:
                                    price_data['last'] = text
                            except:
                                pass
                        
                        # Change %: contains %
                        if '%' in text and ('+' in text or '-' in text):
                            price_data['change_pct'] = text
                        
                        # Change: starts with + or -
                        if text.startswith(('+', '-')) and any(char.isdigit() for char in text) and '%' not in text:
                            price_data['change'] = text
            
            # Build commodity entry
            commodity_data = {
                'name': name,
                'url': href if href.startswith('http') else f"https://www.investing.com{href}" if href else '',
                'category': category,
                'last': price_data.get('last', 'N/A'),
                'change': price_data.get('change', 'N/A'),
                'change_pct': price_data.get('change_pct', 'N/A')
            }
            
            commodities.append(commodity_data)
        
        # Also try to extract from tables more systematically
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Check if table contains commodity data
            table_text = ' '.join([row.get_text() for row in rows[:3]]).lower()
            if any(kw in table_text for kw_list in commodity_keywords.values() for kw in kw_list):
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
                
                # Find column indices
                name_idx = None
                last_idx = None
                change_idx = None
                change_pct_idx = None
                
                for i, header in enumerate(headers):
                    if 'name' in header or 'commodity' in header:
                        name_idx = i
                    if 'last' in header:
                        last_idx = i
                    if 'change' in header and '%' not in header:
                        change_idx = i
                    if 'change' in header and '%' in header or 'chg' in header and '%' in header:
                        change_pct_idx = i
                
                if name_idx is not None:
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) <= name_idx:
                            continue
                        
                        name_cell = cells[name_idx]
                        name_link = name_cell.find('a')
                        if name_link:
                            name = name_link.get_text(strip=True)
                            href = name_link.get('href', '')
                        else:
                            name = name_cell.get_text(strip=True)
                            href = ''
                        
                        if not name or len(name) < 2:
                            continue
                        
                        # Skip if already added
                        if name.lower() in seen_commodities:
                            continue
                        
                        # Only add if it's a commodity
                        name_lower = name.lower()
                        if not any(kw in name_lower for kw_list in commodity_keywords.values() for kw in kw_list):
                            if not href or '/commodities/' not in href.lower():
                                continue
                        
                        seen_commodities.add(name.lower())
                        
                        # Extract price data
                        commodity_data = {
                            'name': name,
                            'url': href if href.startswith('http') else f"https://www.investing.com{href}" if href else '',
                            'last': cells[last_idx].get_text(strip=True) if last_idx and last_idx < len(cells) else 'N/A',
                            'change': cells[change_idx].get_text(strip=True) if change_idx and change_idx < len(cells) else 'N/A',
                            'change_pct': cells[change_pct_idx].get_text(strip=True) if change_pct_idx and change_pct_idx < len(cells) else 'N/A',
                        }
                        
                        # Determine category
                        if any(kw in name_lower for kw in commodity_keywords['metals']):
                            commodity_data['category'] = 'metals'
                        elif any(kw in name_lower for kw in commodity_keywords['energy']):
                            commodity_data['category'] = 'energy'
                        elif any(kw in name_lower for kw in commodity_keywords['agriculture']):
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
