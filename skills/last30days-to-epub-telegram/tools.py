"""
Last30Days → EPUB → Telegram Composite Skill

Uses composite skill framework to combine:
- last30days-claude-cli: Research
- document-converter: EPUB generation
- telegram-sender: Telegram delivery

DRY: Reuses existing skills, no duplication.
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("last30days-to-epub-telegram")


logger = logging.getLogger(__name__)


def _json_to_markdown(data: Dict[str, Any], topic: str) -> str:
    """Convert last30days JSON output to markdown."""
    lines = [
        f"# Research Results: \"{topic}\"",
        f"",
        f"**Date Range:** {data.get('date_range', 'Last 30 days')}",
        f"**Mode:** Jotty Web Search (DuckDuckGo)",
        f"",
    ]
    
    # Reddit section
    if 'reddit' in data and data['reddit']:
        lines.append("## Reddit Discussions")
        lines.append("")
        for item in data['reddit'][:10]:
            lines.append(f"### {item.get('title', 'Untitled')}")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            if item.get('insights'):
                lines.append(f"**Insights:** {item['insights']}")
            lines.append("")
    
    # X section
    if 'x' in data and data['x']:
        lines.append("## X/Twitter Posts")
        lines.append("")
        for item in data['x'][:10]:
            if item.get('text'):
                lines.append(f"**{item.get('author', 'Unknown')}:** {item['text'][:200]}...")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            lines.append("")
    
    # Web section
    if 'web' in data and data['web']:
        lines.append("## Web Sources")
        lines.append("")
        for item in data['web'][:10]:
            lines.append(f"### {item.get('title', 'Untitled')}")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            if item.get('insights'):
                lines.append(f"**Insights:** {item['insights']}")
            lines.append("")
    
    # Patterns
    if 'patterns' in data and data['patterns']:
        lines.append("## Key Patterns")
        lines.append("")
        for pattern in data['patterns']:
            lines.append(f"- {pattern}")
        lines.append("")
    
    # Recommendations
    if 'recommendations' in data and data['recommendations']:
        lines.append("## Recommendations")
        lines.append("")
        for rec in data['recommendations']:
            lines.append(f"- {rec}")
        lines.append("")
    
    return "\n".join(lines)


async def last30days_to_epub_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research topic using last30days, generate EPUB, and send to Telegram.
    
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
            - epub_path (str): Path to generated EPUB
            - telegram_sent (bool): Whether sent to Telegram
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        topic = params.get('topic')
        if not topic:
            return {
                'success': False,
                'error': 'topic parameter is required'
            }
        
        registry = get_skills_registry()
        registry.init()
        
        logger.info(f"🔍 Last30Days → EPUB → Telegram workflow: {topic}")
        
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
                'emit': 'md'  # Request markdown format
            })
        else:
            research_result = research_tool({
                'topic': topic,
                'deep': params.get('deep', False),
                'quick': params.get('quick', False),
                'emit': 'md'  # Request markdown format
            })
        
        if not research_result.get('success'):
            return {
                'success': False,
                'error': f"Research failed: {research_result.get('error')}"
            }
        
        # last30days returns 'output' field, check format
        research_output = research_result.get('output', '')
        output_format = research_result.get('format', 'compact')
        
        # If format is 'md', output is already markdown
        # If format is 'json', need to convert
        # If format is 'compact', need to convert
        if output_format == 'md':
            markdown_content = research_output
        elif output_format == 'json':
            # Convert JSON to markdown
            import json
            data = research_output if isinstance(research_output, dict) else json.loads(research_output)
            markdown_content = _json_to_markdown(data, topic)
        else:  # compact
            # Use compact output as markdown (it's already text)
            markdown_content = research_output
        
        if not markdown_content:
            return {
                'success': False,
                'error': 'No content returned from research'
            }
        
        logger.info(f"✅ Research complete: {len(markdown_content)} chars")
        
        # Step 2: Generate EPUB
        logger.info("📚 Step 2: Generating EPUB...")
        doc_converter_skill = registry.get_skill('document-converter')
        if not doc_converter_skill:
            return {
                'success': False,
                'error': 'document-converter skill not available'
            }
        
        convert_tool = doc_converter_skill.tools.get('convert_to_epub_tool')
        
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
            epub_filename = f"{safe_title}_{timestamp}.epub"
            epub_path = output_dir / epub_filename
            
            # Convert to EPUB
            if inspect.iscoroutinefunction(convert_tool):
                epub_result = await convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(epub_path),
                    'title': title,
                    'author': 'Jotty Last30Days Research'
                })
            else:
                epub_result = convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(epub_path),
                    'title': title,
                    'author': 'Jotty Last30Days Research'
                })
            
            if not epub_result.get('success'):
                return {
                    'success': False,
                    'error': f"EPUB generation failed: {epub_result.get('error')}"
                }
            
            logger.info(f"✅ EPUB generated: {epub_path}")
            
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
                                'file_path': str(epub_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📚 {title}\n\nResearch from last 30 days using Jotty"
                            })
                        else:
                            telegram_result = send_file_tool({
                                'file_path': str(epub_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📚 {title}\n\nResearch from last 30 days using Jotty"
                            })
                        telegram_sent = telegram_result.get('success', False)
                        if telegram_sent:
                            logger.info("✅ Sent to Telegram")
                        else:
                            logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
            
            return {
                'success': True,
                'epub_path': str(epub_path),
                'telegram_sent': telegram_sent,
                'file_size': epub_path.stat().st_size,
                'topic': topic,
                'title': title
            }
        
        finally:
            # Clean up temp file
            if Path(tmp_md_path).exists():
                Path(tmp_md_path).unlink()
        
    except Exception as e:
        logger.error(f"Last30Days → EPUB → Telegram workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['last30days_to_epub_telegram_tool']
