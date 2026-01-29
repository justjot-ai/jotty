"""
V2V to PDF + Telegram + reMarkable Skill

Complete workflow: Search V2V → Generate PDF → Send to Telegram and reMarkable.
"""
import asyncio
import inspect
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def v2v_to_pdf_and_send_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search V2V trending topics, generate PDF, and send to Telegram/reMarkable.
    
    Args:
        params: Dictionary containing:
            - query (str, optional): Search query (default: 'trending topics')
            - title (str, optional): Report title
            - send_telegram (bool, optional): Send to Telegram (default: True)
            - send_remarkable (bool, optional): Send to reMarkable (default: True)
            - telegram_chat_id (str, optional): Telegram chat ID
            - remarkable_folder (str, optional): reMarkable folder (default: '/')
            - output_dir (str, optional): Output directory
    
    Returns:
        Dictionary with:
            - success (bool): Whether workflow succeeded
            - pdf_path (str): Path to generated PDF
            - telegram_sent (bool): Whether sent to Telegram
            - remarkable_sent (bool): Whether sent to reMarkable
            - error (str, optional): Error message if failed
    """
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry
        
        query = params.get('query', 'trending topics')
        title = params.get('title', f'V2V Trending: {query}')
        send_telegram = params.get('send_telegram', True)
        send_remarkable = params.get('send_remarkable', True)
        
        logger.info(f"🔍 V2V to PDF workflow: {query}")
        
        registry = get_skills_registry()
        registry.init()
        
        # Step 1: Search V2V
        logger.info("📡 Step 1: Searching V2V.ai...")
        v2v_skill = registry.get_skill('v2v-trending-search')
        if not v2v_skill:
            return {
                'success': False,
                'error': 'v2v-trending-search skill not available'
            }
        
        search_tool = v2v_skill.tools.get('search_v2v_trending_tool')
        if inspect.iscoroutinefunction(search_tool):
            search_result = await search_tool({
                'query': query,
                'format': 'markdown'
            })
        else:
            search_result = search_tool({
                'query': query,
                'format': 'markdown'
            })
        
        if not search_result.get('success'):
            return {
                'success': False,
                'error': f"V2V search failed: {search_result.get('error')}"
            }
        
        markdown_content = search_result.get('content', '')
        logger.info(f"✅ Found {search_result.get('results_count', 0)} results")
        
        # Step 2: Generate PDF
        logger.info("📄 Step 2: Generating PDF...")
        doc_converter_skill = registry.get_skill('document-converter')
        if not doc_converter_skill:
            return {
                'success': False,
                'error': 'document-converter skill not available'
            }
        
        convert_tool = doc_converter_skill.tools.get('convert_to_pdf_tool')
        
        # Create temp markdown file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(markdown_content)
            tmp_md_path = tmp_file.name
        
        try:
            # Determine output directory
            output_dir = params.get('output_dir')
            if not output_dir:
                current_file = Path(__file__).resolve()
                jotty_dir = current_file.parent.parent.parent
                stock_market_root = jotty_dir.parent
                output_dir = stock_market_root / 'outputs'
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"{safe_title}_{timestamp}.pdf"
            pdf_path = output_dir / pdf_filename
            
            # Convert to PDF (check if async)
            if inspect.iscoroutinefunction(convert_tool):
                pdf_result = await convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty V2V Research',
                    'page_size': 'a4'
                })
            else:
                pdf_result = convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty V2V Research',
                    'page_size': 'a4'
                })
            
            if not pdf_result.get('success'):
                return {
                    'success': False,
                    'error': f"PDF generation failed: {pdf_result.get('error')}"
                }
            
            logger.info(f"✅ PDF generated: {pdf_path}")
            
            # Step 3: Send to Telegram
            telegram_sent = False
            if send_telegram:
                logger.info("📱 Step 3: Sending to Telegram...")
                telegram_skill = registry.get_skill('telegram-sender')
                if telegram_skill:
                    send_file_tool = telegram_skill.tools.get('send_telegram_file_tool')
                    if send_file_tool:
                        telegram_chat_id = params.get('telegram_chat_id')
                        if inspect.iscoroutinefunction(send_file_tool):
                            telegram_result = await send_file_tool({
                                'file_path': str(pdf_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📊 {title}\n\nGenerated from V2V.ai trending search"
                            })
                        else:
                            telegram_result = send_file_tool({
                                'file_path': str(pdf_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📊 {title}\n\nGenerated from V2V.ai trending search"
                            })
                        telegram_sent = telegram_result.get('success', False)
                        if telegram_sent:
                            logger.info("✅ Sent to Telegram")
                        else:
                            logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
                    else:
                        logger.warning("⚠️  send_telegram_file_tool not found")
                else:
                    logger.warning("⚠️  telegram-sender skill not available")
            
            # Step 4: Send to reMarkable
            remarkable_sent = False
            if send_remarkable:
                logger.info("📱 Step 4: Sending to reMarkable...")
                remarkable_skill = registry.get_skill('remarkable-sender')
                if remarkable_skill:
                    send_remarkable_tool = remarkable_skill.tools.get('send_to_remarkable_tool')
                    if send_remarkable_tool:
                        remarkable_folder = params.get('remarkable_folder', '/')
                        if inspect.iscoroutinefunction(send_remarkable_tool):
                            remarkable_result = await send_remarkable_tool({
                                'file_path': str(pdf_path),
                                'folder': remarkable_folder,
                                'document_name': safe_title
                            })
                        else:
                            remarkable_result = send_remarkable_tool({
                                'file_path': str(pdf_path),
                                'folder': remarkable_folder,
                                'document_name': safe_title
                            })
                        remarkable_sent = remarkable_result.get('success', False)
                        if remarkable_sent:
                            logger.info("✅ Sent to reMarkable")
                        else:
                            logger.warning(f"⚠️  reMarkable send failed: {remarkable_result.get('error')}")
                    else:
                        logger.warning("⚠️  send_to_remarkable_tool not found")
                else:
                    logger.warning("⚠️  remarkable-sender skill not available")
            
            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'telegram_sent': telegram_sent,
                'remarkable_sent': remarkable_sent,
                'file_size': pdf_path.stat().st_size,
                'results_count': search_result.get('results_count', 0)
            }
        
        finally:
            # Clean up temp file
            if Path(tmp_md_path).exists():
                Path(tmp_md_path).unlink()
        
    except Exception as e:
        logger.error(f"V2V to PDF workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['v2v_to_pdf_and_send_tool']
