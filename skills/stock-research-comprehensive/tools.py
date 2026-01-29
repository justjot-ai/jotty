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
            - country (str, optional): Country/Exchange (e.g., 'India', 'NSE', 'BSE')
            - exchange (str, optional): Exchange name (e.g., 'NSE', 'BSE', 'NYSE')
            - output_dir (str, optional): Output directory
            - title (str, optional): Report title
            - author (str, optional): Report author
            - page_size (str, optional): PDF page size
            - telegram_chat_id (str, optional): Telegram chat ID
            - send_telegram (bool, optional): Send to Telegram
            - max_results_per_aspect (int, optional): Max results per research aspect
            - target_pages (int, optional): Target report length in pages (default: 10)
    
    Returns:
        Dictionary with research results and file paths
    """
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry
        
        ticker = params.get('ticker', '').upper().strip()
        if not ticker:
            return {
                'success': False,
                'error': 'ticker parameter is required'
            }
        
        company_name = params.get('company_name', ticker)
        country = params.get('country', '')
        exchange = params.get('exchange', '')
        output_dir = params.get('output_dir', os.path.expanduser('~/jotty/reports'))
        
        # Build title with country/exchange info
        location_info = []
        if country:
            location_info.append(country)
        if exchange:
            location_info.append(exchange)
        location_suffix = f" ({', '.join(location_info)})" if location_info else ""
        
        title = params.get('title', f'{company_name} ({ticker}){location_suffix} - Comprehensive Research Report')
        author = params.get('author', 'Jotty Stock Research')
        page_size = params.get('page_size', 'a4')
        telegram_chat_id = params.get('telegram_chat_id')
        send_telegram = params.get('send_telegram', True)
        max_results = params.get('max_results_per_aspect', 15)
        target_pages = params.get('target_pages', 10)
        
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
        
        # Step 1: Comprehensive parallel research on multiple aspects
        logger.info("📊 Step 1: Conducting comprehensive parallel research...")
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        if not search_tool:
            return {
                'success': False,
                'error': 'search_web_tool not found in web-search skill'
            }
        
        # Build location context for searches
        location_context = ""
        if country:
            location_context = f" {country}"
        if exchange:
            location_context += f" {exchange}"
        
        # Prepare comprehensive search queries covering all aspects
        search_queries = {
            'fundamentals': f"{company_name} {ticker}{location_context} fundamentals financial metrics revenue earnings profit margin ROE ROA debt equity",
            'financial_statements': f"{company_name} {ticker}{location_context} financial statements balance sheet cash flow P&L annual report",
            'valuation': f"{company_name} {ticker}{location_context} valuation P/E ratio P/B ratio EV/EBITDA DCF intrinsic value",
            'business_model': f"{company_name} {ticker}{location_context} business model products services market share competitive advantage",
            'industry_analysis': f"{company_name} {ticker}{location_context} industry analysis sector trends market size growth prospects",
            'management': f"{company_name} {ticker}{location_context} management team CEO leadership corporate governance ESG",
            'technicals': f"{company_name} {ticker}{location_context} technical analysis price trends indicators chart patterns support resistance",
            'broker_reports': f"{company_name} {ticker}{location_context} analyst reports ratings price target research recommendations",
            'news_sentiment': f"{company_name} {ticker}{location_context} recent news developments announcements quarterly results",
            'risks': f"{company_name} {ticker}{location_context} risks challenges threats regulatory issues competition",
            'growth_prospects': f"{company_name} {ticker}{location_context} growth prospects expansion plans new products market opportunities",
            'dividend_history': f"{company_name} {ticker}{location_context} dividend history yield payout ratio dividend policy"
        }
        
        # Execute searches in parallel
        async def run_search(query, aspect_name):
            """Run a search and return results."""
            try:
                if inspect.iscoroutinefunction(search_tool):
                    result = await search_tool({'query': query, 'max_results': max_results})
                else:
                    result = search_tool({'query': query, 'max_results': max_results})
                
                if result.get('success'):
                    logger.info(f"✅ {aspect_name}: {result.get('count', 0)} results")
                else:
                    logger.warning(f"⚠️  {aspect_name} failed: {result.get('error')}")
                
                return result
            except Exception as e:
                logger.error(f"❌ {aspect_name} error: {e}")
                return {'success': False, 'error': str(e), 'results': []}
        
        # Run all searches in parallel
        logger.info(f"🔍 Running {len(search_queries)} parallel searches...")
        search_tasks = [
            run_search(query, aspect_name) 
            for aspect_name, query in search_queries.items()
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Organize results by aspect
        research_data = {}
        for (aspect_name, _), result in zip(search_queries.items(), search_results):
            if isinstance(result, Exception):
                research_data[aspect_name] = {'success': False, 'error': str(result), 'results': []}
            else:
                research_data[aspect_name] = result
        
        # Step 2: Combine research using Claude CLI
        logger.info("🤖 Step 2: Synthesizing comprehensive research with Claude...")
        
        # Calculate total results for logging
        total_results = sum(r.get('count', 0) for r in research_data.values())
        total_snippets = sum(min(len(r.get('results', [])), 5) for r in research_data.values())
        logger.info(f"📊 Total research data: {total_results} results, sending top {total_snippets} snippets to Claude")
        
        # Prepare comprehensive research content for Claude
        # Optimized: Top 5 results per aspect, 300 char snippets
        research_content = f"""
# Comprehensive Research Data for {company_name} ({ticker}){location_suffix}

## Company Information
- **Ticker:** {ticker}
- **Company:** {company_name}
- **Location:** {country or 'Not specified'}{f' ({exchange})' if exchange else ''}
- **Report Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Research Sources:** {total_results} total results across {len(search_queries)} research aspects

"""
        
        # Add all research aspects
        aspect_titles = {
            'fundamentals': 'Fundamentals & Financial Metrics',
            'financial_statements': 'Financial Statements & Annual Reports',
            'valuation': 'Valuation Analysis',
            'business_model': 'Business Model & Products',
            'industry_analysis': 'Industry Analysis & Sector Trends',
            'management': 'Management & Corporate Governance',
            'technicals': 'Technical Analysis',
            'broker_reports': 'Broker Research & Analyst Reports',
            'news_sentiment': 'Recent News & Developments',
            'risks': 'Risks & Challenges',
            'growth_prospects': 'Growth Prospects & Opportunities',
            'dividend_history': 'Dividend History & Policy'
        }
        
        for aspect_name, aspect_title in aspect_titles.items():
            research_content += f"\n## {aspect_title}\n\n"
            result = research_data.get(aspect_name, {})
            
            if result.get('success') and result.get('results'):
                # Use top 5 most relevant results per aspect (reduced from 8)
                for i, res in enumerate(result['results'][:5], 1):
                    research_content += f"### {i}. {res.get('title', 'Untitled')}\n"
                    research_content += f"**URL:** {res.get('url', '')}\n"
                    snippet = res.get('snippet', 'No summary available')
                    if snippet:
                        # Limit snippet to 300 chars (reduced from 500) to keep prompt manageable
                        research_content += f"**Summary:** {snippet[:300]}\n\n"
                    else:
                        research_content += "\n"
            else:
                research_content += f"No {aspect_title.lower()} research results available.\n\n"
        
        # Generate comprehensive 10-page report using Claude
        claude_prompt = f"""WRITE A COMPLETE, DETAILED STOCK RESEARCH REPORT for {company_name} ({ticker}){location_suffix} based on the extensive research data provided below.

**CRITICAL: YOU MUST WRITE OUT THE COMPLETE, FULL REPORT CONTENT - NOT A SUMMARY, OVERVIEW, OR DESCRIPTION**

**IMPORTANT REQUIREMENTS:**
- Target length: {target_pages} FULL PAGES of detailed content (approximately 5,000-8,000 words minimum)
- WRITE OUT EVERY SECTION IN COMPLETE DETAIL - each section should be 400-800 words with multiple paragraphs
- Be extremely thorough, detailed, and analytical - write extensive paragraphs of analysis for each subsection
- Include specific numbers, metrics, data points, and dates wherever available from the research data
- Cover EVERY aspect comprehensively with extensive written analysis - don't skip or summarize
- Use professional financial analysis language suitable for institutional investors
- Structure the report with clear sections and subsections using markdown headers
- DO NOT write summaries, overviews, or descriptions - WRITE THE ACTUAL FULL REPORT CONTENT WITH ALL DETAILS
- Expand each point with detailed explanations, analysis, and context

**REPORT STRUCTURE (Expand each section with detailed subsections):**

1. **Executive Summary** (1-1.5 pages)
   - Company overview and business description
   - Key investment highlights
   - Current stock price and market cap
   - Quick snapshot of financials
   - Investment thesis summary

2. **Company Overview & Business Model** (1-1.5 pages)
   - Company history and background
   - Business segments and product portfolio
   - Geographic presence and market position
   - Competitive advantages and moats
   - Strategic initiatives

3. **Industry Analysis & Market Position** (1 page)
   - Industry overview and trends
   - Market size and growth prospects
   - Competitive landscape
   - Industry challenges and opportunities
   - Company's position within the industry

4. **Financial Analysis** (2 pages)
   - Revenue trends and growth drivers
   - Profitability analysis (margins, ROE, ROA)
   - Balance sheet strength
   - Cash flow analysis
   - Key financial ratios and comparisons
   - Historical performance trends

5. **Valuation Analysis** (1-1.5 pages)
   - Current valuation metrics (P/E, P/B, EV/EBITDA)
   - Peer comparison
   - DCF analysis (if data available)
   - Intrinsic value estimation
   - Valuation conclusion

6. **Technical Analysis** (1 page)
   - Price trends and patterns
   - Key support and resistance levels
   - Technical indicators (RSI, MACD, Moving Averages)
   - Chart patterns
   - Trading recommendations

7. **Management & Corporate Governance** (0.5-1 page)
   - Management team overview
   - Corporate governance practices
   - ESG factors
   - Management track record

8. **Broker Research & Analyst Coverage** (1 page)
   - Analyst ratings summary
   - Consensus price targets
   - Key research highlights from major brokers
   - Analyst recommendations breakdown
   - Recent upgrades/downgrades

9. **Risks & Challenges** (1 page)
   - Key business risks
   - Regulatory risks
   - Competitive threats
   - Market risks
   - Company-specific challenges

10. **Growth Prospects & Investment Outlook** (1 page)
    - Growth drivers and catalysts
    - Expansion plans
    - New product launches
    - Market opportunities
    - Future outlook

11. **Dividend Analysis** (0.5 page)
    - Dividend history
    - Current yield
    - Payout ratio
    - Dividend sustainability

12. **Conclusion & Investment Recommendation** (0.5-1 page)
    - Summary of key findings
    - Investment thesis
    - Risk-reward assessment
    - Target price (if applicable)
    - Final recommendation

**RESEARCH DATA:**
{research_content}

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
- WRITE THE COMPLETE, FULL REPORT CONTENT - NOT A SUMMARY OR DESCRIPTION
- Each major section should be 400-800 words with multiple detailed paragraphs
- Executive Summary should be 300-500 words
- Write in a professional, analytical tone suitable for institutional investors
- Include specific numbers, percentages, dates, and metrics from the research data above
- Use tables, bullet points, and structured formatting for clarity
- Be comprehensive - aim for {target_pages} FULL PAGES ({target_pages * 500}-{target_pages * 800} words minimum)
- Synthesize information from multiple sources - don't just list facts, provide deep analytical paragraphs
- Expand each subsection with 3-5 detailed paragraphs explaining the analysis
- If certain data is missing, note it but still provide extensive analysis based on available information
- Ensure the report is thorough enough to serve as a complete investment research document
- Focus on actionable insights and investment implications

**FORMAT REQUIREMENTS:**
- Start immediately with: "# {company_name} ({ticker}){location_suffix} - Comprehensive Research Report"
- Write out ALL 12 sections in FULL DETAIL with complete content
- Each major section should have multiple subsections (###) with detailed paragraphs
- Use markdown formatting (## for sections, ### for subsections, **bold** for emphasis, tables where appropriate)
- Include specific data points, financial metrics, percentages, and analysis throughout
- Write paragraphs, not bullet points (use bullets only for lists within paragraphs)

**WORD COUNT TARGETS PER SECTION:**
1. Executive Summary: 300-500 words
2. Company Overview: 500-700 words  
3. Industry Analysis: 400-600 words
4. Financial Analysis: 600-900 words (most detailed)
5. Valuation Analysis: 500-700 words
6. Technical Analysis: 400-600 words
7. Management: 300-500 words
8. Broker Research: 400-600 words
9. Risks: 400-600 words
10. Growth Prospects: 400-600 words
11. Dividend Analysis: 300-400 words
12. Conclusion: 300-500 words

**NOTE:** The research data above contains summaries from {total_results} search results. 

**START WRITING THE COMPLETE REPORT NOW. BEGIN WITH THE HEADER AND EXECUTIVE SUMMARY, THEN CONTINUE WITH ALL SECTIONS IN FULL DETAIL. DO NOT STOP UNTIL YOU HAVE WRITTEN ALL {target_pages} PAGES OF CONTENT:**"""
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return {
                'success': False,
                'error': 'generate_text_tool not found in claude-cli-llm skill'
            }
        
        # Use longer timeout for comprehensive report generation (10 pages)
        # Increased to 30 minutes (1800s) - generating full detailed report takes time
        timeout_seconds = 1800  # 30 minutes for comprehensive report generation
        
        logger.info(f"⏱️  Using {timeout_seconds//60} minute timeout for Claude generation")
        logger.info(f"📝 Generating comprehensive {target_pages}-page report (target: 5,000-8,000 words)")
        
        if inspect.iscoroutinefunction(generate_tool):
            claude_result = await generate_tool({
                'prompt': claude_prompt,
                'model': 'sonnet',
                'timeout': timeout_seconds
            })
        else:
            claude_result = generate_tool({
                'prompt': claude_prompt,
                'model': 'sonnet',
                'timeout': timeout_seconds
            })
        
        if not claude_result.get('success'):
            return {
                'success': False,
                'error': f'Claude generation failed: {claude_result.get("error")}',
                'research_summary': {k: {'success': v.get('success'), 'count': v.get('count', 0)} 
                                    for k, v in research_data.items()}
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
        
        # Prepare summary of research results
        research_summary = {}
        for aspect_name, result in research_data.items():
            research_summary[aspect_name] = {
                'success': result.get('success', False),
                'count': result.get('count', 0),
                'results': result.get('results', [])[:2]  # Top 2 for summary
            }
        
        return {
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'country': country,
            'exchange': exchange,
            'md_path': str(md_path),
            'pdf_path': pdf_output_path,
            'research_summary': research_summary,
            'total_research_aspects': len(search_queries),
            'total_results': sum(r.get('count', 0) for r in research_data.values()),
            'telegram_sent': telegram_sent,
            'target_pages': target_pages
        }
        
    except Exception as e:
        logger.error(f"Comprehensive stock research error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Research failed: {str(e)}'
        }
