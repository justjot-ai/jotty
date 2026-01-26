"""
Comprehensive Stock Research Skill

Performs research on fundamentals, technicals, and broker reports.
Generates markdown report, converts to PDF, and sends to Telegram.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


async def comprehensive_stock_research_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive stock research and generate PDF report.
    
    Args:
        params: Dictionary containing:
            - ticker (str, required): Stock ticker symbol
            - company_name (str, optional): Full company name
            - output_dir (str, optional): Output directory
            - title (str, optional): Report title
            - author (str, optional): Report author
            - page_size (str, optional): PDF page size
            - telegram_chat_id (str, optional): Telegram chat ID
            - send_telegram (bool, optional): Send to Telegram
            - max_results_per_aspect (int, optional): Max results per research aspect
    
    Returns:
        Dictionary with research results and file paths
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        ticker = params.get('ticker', '').upper().strip()
        if not ticker:
            return {
                'success': False,
                'error': 'ticker parameter is required'
            }
        
        company_name = params.get('company_name', ticker)
        output_dir = params.get('output_dir', os.path.expanduser('~/jotty/reports'))
        title = params.get('title', f'{company_name} ({ticker}) - Comprehensive Research Report')
        author = params.get('author', 'Jotty Stock Research')
        page_size = params.get('page_size', 'a4')
        telegram_chat_id = params.get('telegram_chat_id')
        send_telegram = params.get('send_telegram', True)
        max_results = params.get('max_results_per_aspect', 10)
        
        logger.info(f"🔍 Starting comprehensive research for {company_name} ({ticker})")
        
        # Initialize registry
        registry = get_skills_registry()
        registry.init()
        
        # Get required skills
        web_search_skill = registry.get_skill('web-search')
        claude_skill = registry.get_skill('claude-cli-llm')
        file_ops_skill = registry.get_skill('file-operations')
        doc_converter_skill = registry.get_skill('document-converter')
        telegram_skill = registry.get_skill('telegram-sender')
        
        if not all([web_search_skill, claude_skill, file_ops_skill, doc_converter_skill]):
            return {
                'success': False,
                'error': 'Required skills not available: web-search, claude-cli-llm, file-operations, document-converter'
            }
        
        # Step 1: Parallel research on three aspects
        logger.info("📊 Step 1: Conducting parallel research...")
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        if not search_tool:
            return {
                'success': False,
                'error': 'search_web_tool not found in web-search skill'
            }
        
        # Prepare search queries
        fundamentals_query = f"{company_name} {ticker} fundamentals financial metrics revenue earnings"
        technicals_query = f"{ticker} stock technical analysis price trends indicators chart patterns"
        broker_query = f"{company_name} {ticker} analyst reports ratings price target research"
        
        # Execute searches in parallel
        async def run_search(query, aspect_name):
            """Run a search and return results."""
            if inspect.iscoroutinefunction(search_tool):
                result = await search_tool({'query': query, 'max_results': max_results})
            else:
                result = search_tool({'query': query, 'max_results': max_results})
            
            if result.get('success'):
                logger.info(f"✅ {aspect_name} research: {result.get('count', 0)} results")
            else:
                logger.warning(f"⚠️  {aspect_name} research failed: {result.get('error')}")
            
            return result
        
        # Run all searches in parallel
        fundamentals_result, technicals_result, broker_result = await asyncio.gather(
            run_search(fundamentals_query, 'Fundamentals'),
            run_search(technicals_query, 'Technicals'),
            run_search(broker_query, 'Broker Reports'),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(fundamentals_result, Exception):
            fundamentals_result = {'success': False, 'error': str(fundamentals_result), 'results': []}
        if isinstance(technicals_result, Exception):
            technicals_result = {'success': False, 'error': str(technicals_result), 'results': []}
        if isinstance(broker_result, Exception):
            broker_result = {'success': False, 'error': str(broker_result), 'results': []}
        
        # Step 2: Combine research using Claude CLI
        logger.info("🤖 Step 2: Synthesizing research with Claude...")
        
        # Prepare research content for Claude
        research_content = f"""
# Research Data for {company_name} ({ticker})

## Fundamentals Research
"""
        
        if fundamentals_result.get('success') and fundamentals_result.get('results'):
            for i, result in enumerate(fundamentals_result['results'][:5], 1):
                research_content += f"\n### {i}. {result.get('title', 'Untitled')}\n"
                research_content += f"**URL:** {result.get('url', '')}\n"
                research_content += f"**Summary:** {result.get('snippet', 'No summary available')}\n\n"
        else:
            research_content += "\nNo fundamentals research results available.\n\n"
        
        research_content += "\n## Technical Analysis Research\n"
        
        if technicals_result.get('success') and technicals_result.get('results'):
            for i, result in enumerate(technicals_result['results'][:5], 1):
                research_content += f"\n### {i}. {result.get('title', 'Untitled')}\n"
                research_content += f"**URL:** {result.get('url', '')}\n"
                research_content += f"**Summary:** {result.get('snippet', 'No summary available')}\n\n"
        else:
            research_content += "\nNo technical analysis research results available.\n\n"
        
        research_content += "\n## Broker Research & Analyst Reports\n"
        
        if broker_result.get('success') and broker_result.get('results'):
            for i, result in enumerate(broker_result['results'][:5], 1):
                research_content += f"\n### {i}. {result.get('title', 'Untitled')}\n"
                research_content += f"**URL:** {result.get('url', '')}\n"
                research_content += f"**Summary:** {result.get('snippet', 'No summary available')}\n\n"
        else:
            research_content += "\nNo broker research results available.\n\n"
        
        # Generate comprehensive report using Claude
        claude_prompt = f"""Create a comprehensive stock research report for {company_name} ({ticker}) based on the following research data.

The report should be well-structured markdown with the following sections:

1. **Executive Summary** - Brief overview of the company and key findings
2. **Fundamentals Analysis** - Financial metrics, business model, competitive position, growth prospects
3. **Technical Analysis** - Price trends, technical indicators, support/resistance levels, chart patterns
4. **Broker Research & Analyst Reports** - Analyst ratings, price targets, key research highlights
5. **Conclusion** - Summary of key takeaways and investment considerations

Use the research data provided below. Be thorough, analytical, and professional. Include specific data points and metrics where available.

Research Data:
{research_content}

Generate the complete markdown report now:"""
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return {
                'success': False,
                'error': 'generate_text_tool not found in claude-cli-llm skill'
            }
        
        if inspect.iscoroutinefunction(generate_tool):
            claude_result = await generate_tool({'prompt': claude_prompt, 'model': 'sonnet'})
        else:
            claude_result = generate_tool({'prompt': claude_prompt, 'model': 'sonnet'})
        
        if not claude_result.get('success'):
            return {
                'success': False,
                'error': f'Claude generation failed: {claude_result.get("error")}',
                'fundamentals_research': fundamentals_result,
                'technicals_research': technicals_result,
                'broker_research': broker_result
            }
        
        markdown_content = claude_result.get('text', '')
        
        # Add header with metadata
        header = f"""# {title}

**Ticker:** {ticker}  
**Company:** {company_name}  
**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** {author}

---

"""
        
        full_markdown = header + markdown_content
        
        # Step 3: Write markdown file
        logger.info("📝 Step 3: Writing markdown file...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_ticker = ticker.replace('/', '-').replace('\\', '-')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_filename = f"{safe_ticker}_research_{timestamp}.md"
        md_path = output_path / md_filename
        
        write_tool = file_ops_skill.tools.get('write_file_tool')
        if not write_tool:
            return {
                'success': False,
                'error': 'write_file_tool not found in file-operations skill'
            }
        
        write_result = write_tool({
            'path': str(md_path),
            'content': full_markdown
        })
        
        if not write_result.get('success'):
            return {
                'success': False,
                'error': f'Failed to write markdown: {write_result.get("error")}'
            }
        
        logger.info(f"✅ Markdown saved: {md_path}")
        
        # Step 4: Convert to PDF
        logger.info("📄 Step 4: Converting to PDF...")
        
        pdf_filename = f"{safe_ticker}_research_{timestamp}.pdf"
        pdf_path = output_path / pdf_filename
        
        convert_tool = doc_converter_skill.tools.get('convert_to_pdf_tool')
        if not convert_tool:
            return {
                'success': False,
                'error': 'convert_to_pdf_tool not found in document-converter skill'
            }
        
        if inspect.iscoroutinefunction(convert_tool):
            pdf_result = await convert_tool({
                'input_file': str(md_path),
                'output_file': str(pdf_path),
                'page_size': page_size,
                'title': title,
                'author': author
            })
        else:
            pdf_result = convert_tool({
                'input_file': str(md_path),
                'output_file': str(pdf_path),
                'page_size': page_size,
                'title': title,
                'author': author
            })
        
        if not pdf_result.get('success'):
            return {
                'success': False,
                'error': f'PDF conversion failed: {pdf_result.get("error")}',
                'md_path': str(md_path)
            }
        
        pdf_output_path = pdf_result.get('output_path', str(pdf_path))
        logger.info(f"✅ PDF generated: {pdf_output_path}")
        
        # Step 5: Send to Telegram (optional)
        telegram_sent = False
        if send_telegram and telegram_skill:
            logger.info("📱 Step 5: Sending to Telegram...")
            
            send_tool = telegram_skill.tools.get('send_telegram_file_tool')
            if send_tool:
                caption = f"📊 {title}\n\nTicker: {ticker}\nCompany: {company_name}"
                
                if inspect.iscoroutinefunction(send_tool):
                    telegram_result = await send_tool({
                        'file_path': pdf_output_path,
                        'caption': caption,
                        'chat_id': telegram_chat_id
                    })
                else:
                    telegram_result = send_tool({
                        'file_path': pdf_output_path,
                        'caption': caption,
                        'chat_id': telegram_chat_id
                    })
                
                if telegram_result.get('success'):
                    telegram_sent = True
                    logger.info("✅ PDF sent to Telegram")
                else:
                    logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
        
        return {
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'md_path': str(md_path),
            'pdf_path': pdf_output_path,
            'fundamentals_research': {
                'success': fundamentals_result.get('success'),
                'count': fundamentals_result.get('count', 0),
                'results': fundamentals_result.get('results', [])[:3]  # First 3 for summary
            },
            'technicals_research': {
                'success': technicals_result.get('success'),
                'count': technicals_result.get('count', 0),
                'results': technicals_result.get('results', [])[:3]
            },
            'broker_research': {
                'success': broker_result.get('success'),
                'count': broker_result.get('count', 0),
                'results': broker_result.get('results', [])[:3]
            },
            'telegram_sent': telegram_sent
        }
        
    except Exception as e:
        logger.error(f"Comprehensive stock research error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Research failed: {str(e)}'
        }
