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

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("investing-commodities")


logger = logging.getLogger(__name__)


@tool_wrapper()
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
    status.set_callback(params.pop('_status_callback', None))

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
                            except (ValueError, AttributeError):
                                # Not a valid number, skip
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
        
        # Extract from tables more systematically - this is the main source
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Check if table contains commodity data
            table_text = ' '.join([row.get_text() for row in rows[:5]]).lower()
            has_commodities = any(kw in table_text for kw_list in commodity_keywords.values() for kw in kw_list)
            
            if has_commodities:
                # Get headers from first row
                header_row = rows[0]
                header_cells = header_row.find_all(['th', 'td'])
                headers = [cell.get_text(strip=True).lower() for cell in header_cells]
                
                # Find column indices - be more flexible
                name_idx = None
                last_idx = None
                change_idx = None
                change_pct_idx = None
                
                for i, header in enumerate(headers):
                    header_lower = header.lower().strip()
                    if header_lower == '':
                        continue  # Skip empty headers
                    if ('name' in header_lower or 'commodity' in header_lower) and name_idx is None:
                        name_idx = i
                    if 'last' in header_lower and last_idx is None:
                        last_idx = i
                    if (('change' in header_lower or 'chg' in header_lower) and '%' not in header_lower 
                        and change_idx is None and header_lower not in ['chg.', 'change']):
                        # Make sure it's not the percentage column
                        change_idx = i
                    if (('change' in header_lower or 'chg' in header_lower) and '%' in header_lower) and change_pct_idx is None:
                        change_pct_idx = i
                
                # Debug: log what we found
                logger.debug(f"Table headers: {headers}, name_idx={name_idx}, last_idx={last_idx}, change_idx={change_idx}, change_pct_idx={change_pct_idx}")
                
                # If we found a name column, extract data
                if name_idx is not None:
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) <= name_idx:
                            continue
                        
                        # Extract name - handle empty first column
                        name_cell_idx = name_idx
                        if name_cell_idx >= len(cells):
                            continue
                        
                        name_cell = cells[name_cell_idx]
                        name_link = name_cell.find('a')
                        if name_link:
                            name = name_link.get_text(strip=True)
                            href = name_link.get('href', '')
                        else:
                            name = name_cell.get_text(strip=True)
                            href = ''
                        
                        # Clean name - remove extra text
                        if name:
                            # Remove "derived" and other suffixes
                            name = name.replace('derived', '').strip()
                            # Remove extra whitespace
                            name = ' '.join(name.split())
                            # Take first line if multiline
                            name = name.split('\n')[0].strip()
                            # Remove leading/trailing special chars
                            name = name.strip('•').strip()
                        
                        if not name or len(name) < 2:
                            continue
                        
                        # Skip stock tickers
                        if len(name) <= 5 and name.isupper() and not any(kw in name.lower() for kw_list in commodity_keywords.values() for kw in kw_list):
                            continue
                        
                        # Extract price data FIRST (before duplicate check)
                        last_price = 'N/A'
                        change_val = 'N/A'
                        change_pct_val = 'N/A'
                        
                        # Extract Last price
                        if last_idx is not None and last_idx < len(cells):
                            last_text = cells[last_idx].get_text(strip=True)
                            # Clean up price - remove commas, check if it's a valid number
                            if last_text and last_text != '—' and last_text != '':
                                # Check if it looks like a price (has digits)
                                if any(char.isdigit() for char in last_text):
                                    last_price = last_text.replace(',', '')  # Keep formatting but remove commas for parsing
                        
                        # Extract Change
                        if change_idx is not None and change_idx < len(cells):
                            change_text = cells[change_idx].get_text(strip=True)
                            if change_text and change_text != '—' and change_text != '':
                                # Should start with + or - and contain digits
                                if change_text.startswith(('+', '-')) and any(char.isdigit() for char in change_text):
                                    change_val = change_text
                        
                        # Extract Change %
                        if change_pct_idx is not None and change_pct_idx < len(cells):
                            change_pct_text = cells[change_pct_idx].get_text(strip=True)
                            if change_pct_text and change_pct_text != '—' and change_pct_text != '':
                                if '%' in change_pct_text:
                                    change_pct_val = change_pct_text
                        
                        # If we didn't get change_pct from dedicated column, try to find it in other cells
                        if change_pct_val == 'N/A' and change_pct_idx is None:
                            for i, cell in enumerate(cells):
                                if i == name_idx or i == last_idx:
                                    continue
                                cell_text = cell.get_text(strip=True)
                                if cell_text and '%' in cell_text and ('+' in cell_text or '-' in cell_text):
                                    change_pct_val = cell_text
                                    break
                        
                        # If we have change but not change_pct, try to calculate or find it
                        if change_val != 'N/A' and change_pct_val == 'N/A':
                            # Look for percentage in nearby cells
                            for i in range(max(0, change_idx-1 if change_idx else 0), min(len(cells), (change_idx+3 if change_idx else len(cells)))):
                                if i == name_idx:
                                    continue
                                cell_text = cells[i].get_text(strip=True)
                                if cell_text and '%' in cell_text:
                                    change_pct_val = cell_text
                                    break
                        
                        # Check if already added - if so, update with price data instead of skipping
                        existing_idx = None
                        for idx, existing in enumerate(commodities):
                            if existing.get('name', '').lower() == name.lower():
                                existing_idx = idx
                                break
                        
                        if existing_idx is not None:
                            # Update existing entry with price data
                            existing = commodities[existing_idx]
                            # Only update if we have better data
                            if last_price != 'N/A' and existing.get('last') == 'N/A':
                                existing['last'] = last_price
                            if change_val != 'N/A' and existing.get('change') == 'N/A':
                                existing['change'] = change_val
                            if change_pct_val != 'N/A' and existing.get('change_pct') == 'N/A':
                                existing['change_pct'] = change_pct_val
                            continue  # Skip adding duplicate
                        
                        # Skip if already seen (but not in commodities list)
                        if name.lower() in seen_commodities:
                            continue
                        
                        # Only add if it's a commodity
                        name_lower = name.lower()
                        is_commodity = any(kw in name_lower for kw_list in commodity_keywords.values() for kw in kw_list)
                        if not is_commodity:
                            if not href or '/commodities/' not in href.lower():
                                continue
                        
                        seen_commodities.add(name.lower())
                        
                        # Check if already added - if so, update with price data instead of skipping
                        existing_idx = None
                        for idx, existing in enumerate(commodities):
                            if existing.get('name', '').lower() == name.lower():
                                existing_idx = idx
                                break
                        
                        if existing_idx is not None:
                            # Update existing entry with price data
                            existing = commodities[existing_idx]
                            # Only update if we have better data
                            if last_price != 'N/A' and existing.get('last') == 'N/A':
                                existing['last'] = last_price
                            if change_val != 'N/A' and existing.get('change') == 'N/A':
                                existing['change'] = change_val
                            if change_pct_val != 'N/A' and existing.get('change_pct') == 'N/A':
                                existing['change_pct'] = change_pct_val
                            continue  # Skip adding duplicate
                        
                        # Skip if already seen (but not in commodities list)
                        if name.lower() in seen_commodities:
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
                        
                        commodity_data = {
                            'name': name,
                            'url': href if href.startswith('http') else f"https://www.investing.com{href}" if href else '',
                            'last': last_price,
                            'change': change_val,
                            'change_pct': change_pct_val,
                            'category': category
                        }
                        
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
        elif output_format == 'html':
            # HTML format - professional broker watchlist style
            formatted_output = f"""<b>📈 COMMODITIES WATCHLIST</b>
<i>🕐 {timestamp}</i>

"""
            
            # Summary
            up_count = sum(1 for c in commodities if c.get('change_pct', '').startswith('+'))
            down_count = sum(1 for c in commodities if c.get('change_pct', '').startswith('-'))
            neutral_count = len(commodities) - up_count - down_count
            
            formatted_output += f"""<b>📊 Summary:</b> <code>{len(commodities)}</code> commodities | 
🟢 <b>{up_count}</b> up | 🔴 <b>{down_count}</b> down | ⚪ <b>{neutral_count}</b> neutral

"""
            
            # Group by category
            by_category = {}
            for c in commodities:
                cat = c.get('category', 'other')
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(c)
            
            category_emojis = {
                'energy': '⚡',
                'metals': '🥇',
                'agriculture': '🌾',
                'other': '📊'
            }
            
            for cat in ['energy', 'metals', 'agriculture', 'other']:
                if cat in by_category and by_category[cat]:
                    emoji = category_emojis.get(cat, '📊')
                    formatted_output += f"\n<b>{emoji} {cat.upper()}</b>\n"
                    formatted_output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    
                    # Sort by change % (highest first)
                    sorted_commodities = sorted(
                        by_category[cat],
                        key=lambda x: (
                            float(x.get('change_pct', '0%').replace('%', '').replace('+', '').replace('-', '') or 0)
                            if x.get('change_pct', 'N/A') != 'N/A' and '%' in str(x.get('change_pct', ''))
                            else -999
                        ),
                        reverse=True
                    )
                    
                    for c in sorted_commodities:
                        name = c.get('name', 'N/A')
                        last = c.get('last', 'N/A')
                        change = c.get('change', 'N/A')
                        change_pct = c.get('change_pct', 'N/A')
                        
                        # Clean values
                        if last in ['N/A', '—', '']:
                            last_display = '—'
                        else:
                            last_display = last
                        
                        if change in ['N/A', '—', '']:
                            change_display = ''
                        else:
                            change_display = change
                        
                        if change_pct in ['N/A', '—', '']:
                            change_pct_display = ''
                            color_emoji = ''
                        else:
                            change_pct_display = change_pct
                            if change_pct.startswith('+'):
                                color_emoji = '🟢'
                            elif change_pct.startswith('-'):
                                color_emoji = '🔴'
                            else:
                                color_emoji = '⚪'
                        
                        # Format as compact watchlist entry (broker style - all on one line)
                        # Truncate long names
                        name_display = name[:20] if len(name) > 20 else name
                        
                        # Build the line
                        line = f"<b>{name_display:<20}</b>"
                        line += f" <code>{last_display:>10}</code>"
                        
                        if change_display:
                            line += f" <code>{change_display:>8}</code>"
                        
                        if change_pct_display:
                            line += f" {color_emoji} <b>{change_pct_display}</b>"
                        
                        formatted_output += line + "\n"
                    
                    formatted_output += "\n"
        elif output_format == 'markdown':
            # Group by category
            by_category = {}
            for c in commodities:
                cat = c.get('category', 'other')
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(c)
            
            # Category emojis and headers
            category_info = {
                'energy': {'emoji': '⚡', 'title': 'Energy'},
                'metals': {'emoji': '🥇', 'title': 'Metals'},
                'agriculture': {'emoji': '🌾', 'title': 'Agriculture'},
                'other': {'emoji': '📊', 'title': 'Other'}
            }
            
            # Build formatted output
            formatted_output = f"📈 *Commodities Prices*\n"
            formatted_output += f"🕐 {timestamp}\n\n"
            
            # Summary stats
            total_count = len(commodities)
            up_count = sum(1 for c in commodities if c.get('change_pct', '').startswith('+'))
            down_count = sum(1 for c in commodities if c.get('change_pct', '').startswith('-'))
            
            formatted_output += f"📊 *Summary:* {total_count} commodities | "
            formatted_output += f"📈 {up_count} up | 📉 {down_count} down\n\n"
            formatted_output += "─" * 40 + "\n\n"
            
            # Format each category
            for cat in ['energy', 'metals', 'agriculture', 'other']:
                if cat in by_category and by_category[cat]:
                    info = category_info.get(cat, {'emoji': '📊', 'title': cat.title()})
                    formatted_output += f"{info['emoji']} *{info['title']}*\n\n"
                    
                    # Sort by change % (highest first)
                    sorted_commodities = sorted(
                        by_category[cat],
                        key=lambda x: (
                            float(x.get('change_pct', '0%').replace('%', '').replace('+', '').replace('-', '') or 0)
                            if x.get('change_pct', 'N/A') != 'N/A' and '%' in str(x.get('change_pct', ''))
                            else -999
                        ),
                        reverse=True
                    )
                    
                    # Format as clean list (better for Telegram)
                    for c in sorted_commodities:
                        name = c.get('name', 'N/A')
                        last = c.get('last', 'N/A')
                        change = c.get('change', 'N/A')
                        change_pct = c.get('change_pct', 'N/A')
                        
                        # Clean up values
                        if last == 'N/A' or not last or last.strip() == '':
                            last = '—'
                        if change == 'N/A' or not change or change.strip() == '':
                            change = '—'
                        if change_pct == 'N/A' or not change_pct or change_pct.strip() == '':
                            change_pct = '—'
                        
                        # Format change indicator
                        if change_pct != '—' and change_pct != 'N/A':
                            if change_pct.startswith('+'):
                                indicator = '📈'
                            elif change_pct.startswith('-'):
                                indicator = '📉'
                            else:
                                indicator = '➡️'
                        else:
                            indicator = '➡️'
                        
                        # Format the line
                        formatted_output += f"• *{name}*\n"
                        formatted_output += f"  💵 {last}"
                        if change != '—' and change != 'N/A':
                            formatted_output += f" | {change}"
                        if change_pct != '—' and change_pct != 'N/A':
                            formatted_output += f" {indicator} {change_pct}"
                        formatted_output += "\n\n"
                    
                    formatted_output += "─" * 40 + "\n\n"
        else:  # text
            formatted_output = f"📈 Commodities Prices\n"
            formatted_output += f"🕐 {timestamp}\n\n"
            for c in commodities:
                name = c.get('name', 'N/A')
                last = c.get('last', 'N/A')
                change_pct = c.get('change_pct', 'N/A')
                formatted_output += f"• {name}: {last} ({change_pct})\n"
        
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
