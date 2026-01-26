"""
Last30Days → PDF → Telegram Composite Skill

Uses composite skill framework to combine:
- last30days-claude-cli: Research
- document-converter: PDF generation
- telegram-sender: Telegram delivery

DRY: Reuses existing skills, no duplication.
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def last30days_to_pdf_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research topic using last30days, generate PDF, and send to Telegram.
    
    Args:
        params: Dictionary containing:
            - topic (str, required): Research topic
            - deep (bool, optional): Deep research mode
            - quick (bool, optional): Quick research mode
            - title (str, optional): Report title
            - send_telegram (bool, optional): Send to Telegram (default: True)
            - telegram_chat_id (str, optional): Telegram chat ID
            - output_dir (str, optional): Output directory
    
    Returns:
        Dictionary with:
            - success (bool): Whether workflow succeeded
            - pdf_path (str): Path to generated PDF
            - telegram_sent (bool): Whether sent to Telegram
            - error (str, optional): Error message if failed
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        topic = params.get('topic')
        if not topic:
            return {
                'success': False,
                'error': 'topic parameter is required'
            }
        
        registry = get_skills_registry()
        registry.init()
        
        logger.info(f"🔍 Last30Days → PDF → Telegram workflow: {topic}")
        
        # Step 1: Research using last30days-claude-cli
        logger.info("📡 Step 1: Researching with last30days...")
        last30days_skill = registry.get_skill('last30days-claude-cli')
        if not last30days_skill:
            return {
                'success': False,
                'error': 'last30days-claude-cli skill not available'
            }
        
        research_tool = last30days_skill.tools.get('last30days_claude_cli_tool')
        if not research_tool:
            return {
                'success': False,
                'error': 'last30days_claude_cli_tool not found'
            }
        
        # Execute research
        import inspect
        if inspect.iscoroutinefunction(research_tool):
            research_result = await research_tool({
                'topic': topic,
                'deep': params.get('deep', False),
                'quick': params.get('quick', False),
                'output_format': 'markdown'
            })
        else:
            research_result = research_tool({
                'topic': topic,
                'deep': params.get('deep', False),
                'quick': params.get('quick', False),
                'output_format': 'markdown'
            })
        
        if not research_result.get('success'):
            return {
                'success': False,
                'error': f"Research failed: {research_result.get('error')}"
            }
        
        markdown_content = research_result.get('content', research_result.get('markdown', ''))
        if not markdown_content:
            return {
                'success': False,
                'error': 'No content returned from research'
            }
        
        logger.info(f"✅ Research complete: {len(markdown_content)} chars")
        
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
            title = params.get('title', f'Last30Days: {topic}')
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"{safe_title}_{timestamp}.pdf"
            pdf_path = output_dir / pdf_filename
            
            # Convert to PDF
            if inspect.iscoroutinefunction(convert_tool):
                pdf_result = await convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty Last30Days Research',
                    'page_size': 'a4'
                })
            else:
                pdf_result = convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty Last30Days Research',
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
            if params.get('send_telegram', True):
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
                                'caption': f"📊 {title}\n\nResearch from last 30 days using Jotty"
                            })
                        else:
                            telegram_result = send_file_tool({
                                'file_path': str(pdf_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📊 {title}\n\nResearch from last 30 days using Jotty"
                            })
                        telegram_sent = telegram_result.get('success', False)
                        if telegram_sent:
                            logger.info("✅ Sent to Telegram")
                        else:
                            logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
            
            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'telegram_sent': telegram_sent,
                'file_size': pdf_path.stat().st_size,
                'topic': topic,
                'title': title
            }
        
        finally:
            # Clean up temp file
            if Path(tmp_md_path).exists():
                Path(tmp_md_path).unlink()
        
    except Exception as e:
        logger.error(f"Last30Days → PDF → Telegram workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['last30days_to_pdf_telegram_tool']
