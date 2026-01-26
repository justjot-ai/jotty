"""
Investing.com Commodities to Telegram Skill

Fetches commodities prices and sends to Telegram.
"""
import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directory to path to import other skills
current_dir = Path(__file__).parent
jotty_root = current_dir.parent.parent
sys.path.insert(0, str(jotty_root))

from core.registry.skills_registry import get_skills_registry

logger = logging.getLogger(__name__)


async def commodities_to_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch commodities prices from investing.com and send to Telegram.
    
    Args:
        params: Dictionary containing:
            - category (str, optional): Category filter - 'energy', 'metals', 'agriculture', or 'all' (default: 'all')
            - send_telegram (bool, optional): Whether to send to Telegram (default: True)
            - telegram_chat_id (str, optional): Telegram chat ID (uses env var if not provided)
            - format (str, optional): Message format - 'markdown' or 'text' (default: 'markdown')
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - commodities (list): List of commodities fetched
            - telegram_sent (bool): Whether sent to Telegram
            - telegram_message_id (int, optional): Telegram message ID if sent
            - error (str, optional): Error message if failed
    """
    try:
        registry = get_skills_registry()
        if not registry.initialized:
            registry.init()
        
        category = params.get('category', 'all')
        send_telegram = params.get('send_telegram', True)
        message_format = params.get('format', 'markdown')
        
        # Step 1: Fetch commodities prices
        logger.info(f"Fetching commodities prices (category: {category})...")
        commodities_skill = registry.get_skill('investing-commodities')
        if not commodities_skill:
            return {
                'success': False,
                'error': 'investing-commodities skill not found'
            }
        
        get_prices_tool = commodities_skill.tools.get('get_commodities_prices_tool')
        if not get_prices_tool:
            return {
                'success': False,
                'error': 'get_commodities_prices_tool not found'
            }
        
        prices_result = get_prices_tool({
            'category': category,
            'format': message_format
        })
        
        if not prices_result.get('success'):
            return {
                'success': False,
                'error': f"Failed to fetch prices: {prices_result.get('error')}",
                'commodities': []
            }
        
        commodities = prices_result.get('commodities', [])
        formatted_output = prices_result.get('formatted_output', '')
        timestamp = prices_result.get('timestamp', '')
        
        logger.info(f"Fetched {len(commodities)} commodities")
        
        # Step 2: Send to Telegram
        telegram_sent = False
        telegram_message_id = None
        
        if send_telegram:
            logger.info("Sending to Telegram...")
            telegram_skill = registry.get_skill('telegram-sender')
            if telegram_skill:
                send_message_tool = telegram_skill.tools.get('send_telegram_message_tool')
                if send_message_tool:
                    telegram_chat_id = params.get('telegram_chat_id')
                    
                    # Prepare message
                    message = formatted_output
                    if len(message) > 4096:  # Telegram message limit
                        # Truncate intelligently - keep summary and first few categories
                        lines = formatted_output.split('\n')
                        truncated_lines = []
                        char_count = 0
                        for line in lines:
                            if char_count + len(line) + 1 > 4000:
                                break
                            truncated_lines.append(line)
                            char_count += len(line) + 1
                        message = '\n'.join(truncated_lines)
                        message += f"\n\n... ({len(commodities)} commodities total)"
                    
                    import inspect
                    if inspect.iscoroutinefunction(send_message_tool):
                        telegram_result = await send_message_tool({
                            'message': message,
                            'chat_id': telegram_chat_id,
                            'parse_mode': 'Markdown' if message_format == 'markdown' else None
                        })
                    else:
                        telegram_result = send_message_tool({
                            'message': message,
                            'chat_id': telegram_chat_id,
                            'parse_mode': 'Markdown' if message_format == 'markdown' else None
                        })
                    
                    telegram_sent = telegram_result.get('success', False)
                    telegram_message_id = telegram_result.get('message_id')
                    
                    if telegram_sent:
                        logger.info(f"✅ Sent to Telegram (message_id: {telegram_message_id})")
                    else:
                        logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
                else:
                    logger.warning("⚠️  send_telegram_message_tool not found")
            else:
                logger.warning("⚠️  telegram-sender skill not available")
        
        return {
            'success': True,
            'commodities': commodities,
            'count': len(commodities),
            'timestamp': timestamp,
            'telegram_sent': telegram_sent,
            'telegram_message_id': telegram_message_id,
            'formatted_output': formatted_output
        }
        
    except Exception as e:
        logger.error(f"Error in commodities_to_telegram_tool: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to fetch and send commodities: {str(e)}',
            'commodities': [],
            'telegram_sent': False
        }


__all__ = ['commodities_to_telegram_tool']
