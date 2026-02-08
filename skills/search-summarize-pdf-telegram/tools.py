"""
Search → Summarize → PDF → Telegram Composite Skill

Uses composite skill framework to combine:
- web-search: Source (data retrieval)
- claude-cli-llm: Processor (summarization)
- document-converter: Processor (format conversion)
- telegram-sender: Sink (output delivery)

DRY: Reuses existing skills, no duplication.
Follows Source → Processor → Sink pattern.
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("search-summarize-pdf-telegram")


logger = logging.getLogger(__name__)


def _format_search_results_for_summary(results: list, topic: str) -> str:
    """Format search results into text for summarization."""
    lines = [
        f"# Search Results: {topic}",
        f"",
        f"The following are web search results about '{topic}':",
        f"",
    ]
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        snippet = result.get('snippet', '')
        
        lines.append(f"## Result {i}: {title}")
        if url:
            lines.append(f"URL: {url}")
        if snippet:
            lines.append(f"Summary: {snippet}")
        lines.append("")
    
    return "\n".join(lines)


async def search_summarize_pdf_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search topic, summarize with Claude CLI LLM, generate PDF, and send to Telegram.
    
    Args:
        params: Dictionary containing:
            - topic (str, required): Search topic
            - max_results (int, optional): Max search results (default: 10)
            - summarize_prompt (str, optional): Custom summarization prompt
            - title (str, optional): Report title
            - send_telegram (bool, optional): Send to Telegram (default: True)
            - telegram_chat_id (str, optional): Telegram chat ID
            - output_dir (str, optional): Output directory
            - model (str, optional): Claude model (default: 'sonnet')
    
    Returns:
        Dictionary with:
            - success (bool): Whether workflow succeeded
            - pdf_path (str): Path to generated PDF
            - telegram_sent (bool): Whether sent to Telegram
            - summary (str): Generated summary text
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
        
        logger.info(f"🔍 Search → Summarize → PDF → Telegram workflow: {topic}")
        
        # Step 1: Source - Search web
        logger.info("📡 Step 1: Searching web...")
        web_search_skill = registry.get_skill('web-search')
        if not web_search_skill:
            return {
                'success': False,
                'error': 'web-search skill not available'
            }
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        if not search_tool:
            return {
                'success': False,
                'error': 'search_web_tool not found'
            }
        
        # Execute search
        import inspect
        max_results = params.get('max_results', 10)
        if inspect.iscoroutinefunction(search_tool):
            search_result = await search_tool({
                'query': topic,
                'max_results': max_results
            })
        else:
            search_result = search_tool({
                'query': topic,
                'max_results': max_results
            })
        
        if not search_result.get('success'):
            return {
                'success': False,
                'error': f"Search failed: {search_result.get('error')}"
            }
        
        results = search_result.get('results', [])
        if not results:
            return {
                'success': False,
                'error': 'No search results found'
            }
        
        logger.info(f"✅ Search complete: {len(results)} results")
        
        # Step 2: Processor - Summarize with Claude CLI LLM
        logger.info("🤖 Step 2: Summarizing with Claude CLI...")
        claude_skill = registry.get_skill('claude-cli-llm')
        if not claude_skill:
            return {
                'success': False,
                'error': 'claude-cli-llm skill not available'
            }
        
        summarize_tool = claude_skill.tools.get('summarize_text_tool')
        if not summarize_tool:
            return {
                'success': False,
                'error': 'summarize_text_tool not found'
            }
        
        # Format search results for summarization
        search_content = _format_search_results_for_summary(results, topic)
        
        # Build summarization prompt
        default_prompt = f"Summarize the following search results about '{topic}'. Provide a comprehensive summary that highlights key points, trends, and insights. Format the summary in clear sections with bullet points where appropriate."
        summarize_prompt = params.get('summarize_prompt', default_prompt)
        
        # Execute summarization
        model = params.get('model', 'sonnet')
        if inspect.iscoroutinefunction(summarize_tool):
            summary_result = await summarize_tool({
                'content': search_content,
                'prompt': summarize_prompt,
                'model': model
            })
        else:
            summary_result = summarize_tool({
                'content': search_content,
                'prompt': summarize_prompt,
                'model': model
            })
        
        if not summary_result.get('success'):
            return {
                'success': False,
                'error': f"Summarization failed: {summary_result.get('error')}"
            }
        
        summary_text = summary_result.get('summary', '')
        if not summary_text:
            return {
                'success': False,
                'error': 'No summary generated'
            }
        
        logger.info(f"✅ Summary complete: {len(summary_text)} chars")
        
        # Step 3: Processor - Generate PDF
        logger.info("📄 Step 3: Generating PDF...")
        doc_converter_skill = registry.get_skill('document-converter')
        if not doc_converter_skill:
            return {
                'success': False,
                'error': 'document-converter skill not available'
            }
        
        convert_tool = doc_converter_skill.tools.get('convert_to_pdf_tool')
        
        # Create temp markdown file with summary
        # Add title and metadata
        title = params.get('title', f'Summary: {topic}')
        markdown_content = f"# {title}\n\n"
        markdown_content += f"**Topic:** {topic}\n"
        markdown_content += f"**Sources:** {len(results)} search results\n"
        markdown_content += f"**Model:** {model}\n\n"
        markdown_content += "---\n\n"
        markdown_content += summary_text
        
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
            
            # Convert to PDF
            if inspect.iscoroutinefunction(convert_tool):
                pdf_result = await convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty Search Summary',
                    'page_size': 'a4'
                })
            else:
                pdf_result = convert_tool({
                    'input_file': tmp_md_path,
                    'output_file': str(pdf_path),
                    'title': title,
                    'author': 'Jotty Search Summary',
                    'page_size': 'a4'
                })
            
            if not pdf_result.get('success'):
                return {
                    'success': False,
                    'error': f"PDF generation failed: {pdf_result.get('error')}"
                }
            
            logger.info(f"✅ PDF generated: {pdf_path}")
            
            # Step 4: Sink - Send to Telegram
            telegram_sent = False
            if params.get('send_telegram', True):
                logger.info("📱 Step 4: Sending to Telegram...")
                telegram_skill = registry.get_skill('telegram-sender')
                if telegram_skill:
                    send_file_tool = telegram_skill.tools.get('send_telegram_file_tool')
                    if send_file_tool:
                        telegram_chat_id = params.get('telegram_chat_id')
                        if inspect.iscoroutinefunction(send_file_tool):
                            telegram_result = await send_file_tool({
                                'file_path': str(pdf_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📊 {title}\n\nAI-generated summary using Claude CLI"
                            })
                        else:
                            telegram_result = send_file_tool({
                                'file_path': str(pdf_path),
                                'chat_id': telegram_chat_id,
                                'caption': f"📊 {title}\n\nAI-generated summary using Claude CLI"
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
                'summary': summary_text,
                'file_size': pdf_path.stat().st_size,
                'topic': topic,
                'title': title,
                'search_results_count': len(results)
            }
        
        finally:
            # Clean up temp file
            if Path(tmp_md_path).exists():
                Path(tmp_md_path).unlink()
        
    except Exception as e:
        logger.error(f"Search → Summarize → PDF → Telegram workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['search_summarize_pdf_telegram_tool']
