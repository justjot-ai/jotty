"""
Raffle Winner Picker Skill - Pick random winners from lists or files.

Uses cryptographically secure random selection for fair, unbiased
selection of winners for giveaways and contests.
"""
import asyncio
import logging
import secrets
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("raffle-winner-picker")


logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available, CSV reading will be limited")


@async_tool_wrapper()
async def pick_raffle_winner_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick random winner(s) from a list or file.
    
    Args:
        params:
            - source (str/list): Source file path, URL, or list
            - num_winners (int, optional): Number of winners
            - exclude (list, optional): Entries to exclude
            - weighted_column (str, optional): Column for weighting
            - output_file (str, optional): Path to save results
    
    Returns:
        Dictionary with winners, selection method, timestamp
    """
    status.set_callback(params.pop('_status_callback', None))

    source = params.get('source', '')
    num_winners = params.get('num_winners', 1)
    exclude = params.get('exclude', [])
    weighted_column = params.get('weighted_column', None)
    output_file = params.get('output_file', None)
    
    if not source:
        return {
            'success': False,
            'error': 'source is required'
        }
    
    try:
        # Parse source
        entries = []
        
        if isinstance(source, list):
            # Direct list
            entries = source
        elif isinstance(source, str):
            # File path or URL
            source_path = Path(os.path.expanduser(source))
            
            if source_path.exists():
                # Read from file
                if source_path.suffix.lower() == '.csv':
                    entries = await _read_csv_entries(source_path)
                elif source_path.suffix.lower() in ['.xlsx', '.xls']:
                    entries = await _read_excel_entries(source_path)
                else:
                    # Plain text file
                    entries = source_path.read_text(encoding='utf-8').strip().split('\n')
            else:
                # Assume it's a list string
                entries = [s.strip() for s in source.split(',')]
        
        if not entries:
            return {
                'success': False,
                'error': 'No entries found in source'
            }
        
        # Filter out excluded entries
        if exclude:
            entries = [e for e in entries if e not in exclude]
        
        if len(entries) < num_winners:
            return {
                'success': False,
                'error': f'Not enough entries ({len(entries)}) for {num_winners} winners'
            }
        
        # Select winners using cryptographically secure random
        winners = []
        remaining_entries = entries.copy()
        
        for _ in range(num_winners):
            if not remaining_entries:
                break
            
            # Secure random selection
            winner_index = secrets.randbelow(len(remaining_entries))
            winner = remaining_entries.pop(winner_index)
            
            # Format winner details
            if isinstance(winner, dict):
                winner_info = winner
            else:
                winner_info = {
                    'entry': str(winner),
                    'name': str(winner),
                    'selected_at': datetime.now().isoformat()
                }
            
            winners.append(winner_info)
        
        # Generate results
        timestamp = datetime.now().isoformat()
        selection_method = 'cryptographically_secure_random'
        
        result = {
            'success': True,
            'winners': winners,
            'total_entries': len(entries),
            'selection_method': selection_method,
            'timestamp': timestamp
        }
        
        # Save results if requested
        if output_file:
            await _save_results(result, output_file)
            result['output_file'] = output_file
        
        return result
        
    except Exception as e:
        logger.error(f"Raffle selection failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _read_csv_entries(file_path: Path) -> List[Dict]:
    """Read entries from CSV file."""
    
    entries = []
    
    if PANDAS_AVAILABLE:
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame rows to dicts
            entries = df.to_dict('records')
        except Exception as e:
            logger.debug(f"Pandas CSV read failed: {e}")
    
    # Fallback to standard CSV
    if not entries:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        except Exception as e:
            logger.debug(f"CSV read failed: {e}")
            # Last resort: read as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                entries = [{'entry': line.strip()} for line in f if line.strip()]
    
    return entries


async def _read_excel_entries(file_path: Path) -> List[Dict]:
    """Read entries from Excel file."""
    
    if not PANDAS_AVAILABLE:
        return []
    
    try:
        df = pd.read_excel(file_path)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Excel read failed: {e}")
        return []


async def _save_results(result: Dict, output_file: str) -> None:
    """Save results to file."""
    
    output_path = Path(os.path.expanduser(output_file))
    
    content = f"""# Raffle Winner Selection Results

**Selection Method:** {result['selection_method']}
**Timestamp:** {result['timestamp']}
**Total Entries:** {result['total_entries']}
**Number of Winners:** {len(result['winners'])}

## Winners

"""
    
    for i, winner in enumerate(result['winners'], 1):
        content += f"### Winner {i}\n\n"
        for key, value in winner.items():
            content += f"- **{key}**: {value}\n"
        content += "\n"
    
    output_path.write_text(content, encoding='utf-8')
