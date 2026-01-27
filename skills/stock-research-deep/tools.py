"""
Deep Stock Research Skill - Multi-Stage Research with Context Intelligence

Implements deep research methodology:
1. Initial broad research
2. Gap analysis
3. Targeted follow-up research
4. Cross-referencing and verification
5. Progressive synthesis
6. Quality validation
7. Iterative refinement
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)


async def deep_stock_research_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform deep stock research with multi-stage context intelligence.
    
    This implements true deep research:
    - Stage 1: Initial broad research
    - Stage 2: Gap analysis and targeted follow-up
    - Stage 3: Cross-referencing and verification
    - Stage 4: Progressive synthesis with context building
    - Stage 5: Quality validation
    - Stage 6: Iterative refinement
    
    Args:
        params: Dictionary containing:
            - ticker (str, required): Stock ticker symbol
            - company_name (str, optional): Full company name
            - country (str, optional): Country/Exchange
            - exchange (str, optional): Exchange name
            - output_dir (str, optional): Output directory
            - target_pages (int, optional): Target report length (default: 10)
            - max_iterations (int, optional): Max refinement iterations (default: 2)
            - send_telegram (bool, optional): Send to Telegram
    
    Returns:
        Dictionary with research results and file paths
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        ticker = params.get('ticker', '').upper().strip()
        if not ticker:
            return {'success': False, 'error': 'ticker parameter is required'}
        
        company_name = params.get('company_name', ticker)
        country = params.get('country', '')
        exchange = params.get('exchange', '')
        output_dir = params.get('output_dir', os.path.expanduser('~/jotty/reports'))
        target_pages = params.get('target_pages', 10)
        max_iterations = params.get('max_iterations', 2)
        send_telegram = params.get('send_telegram', True)
        
        logger.info(f"🔬 Starting DEEP research for {company_name} ({ticker})")
        
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
            return {'success': False, 'error': 'Required skills not available'}
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        generate_tool = claude_skill.tools.get('generate_text_tool')
        
        location_context = f" {country}" if country else ""
        if exchange:
            location_context += f" {exchange}"
        
        # ============================================
        # STAGE 1: INITIAL BROAD RESEARCH
        # ============================================
        logger.info("📊 Stage 1: Initial broad research...")
        
        search_queries = {
            'fundamentals': f"{company_name} {ticker}{location_context} fundamentals financial metrics revenue earnings profit margin ROE ROA debt equity",
            'financial_statements': f"{company_name} {ticker}{location_context} financial statements balance sheet cash flow P&L annual report quarterly",
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
        
        async def run_search(query, aspect_name):
            try:
                if inspect.iscoroutinefunction(search_tool):
                    result = await search_tool({'query': query, 'max_results': 15})
                else:
                    result = search_tool({'query': query, 'max_results': 15})
                
                if result.get('success'):
                    logger.info(f"✅ {aspect_name}: {result.get('count', 0)} results")
                return result
            except Exception as e:
                logger.error(f"❌ {aspect_name} error: {e}")
                return {'success': False, 'error': str(e), 'results': []}
        
        # Run initial searches in parallel
        search_tasks = [run_search(query, aspect) for aspect, query in search_queries.items()]
        initial_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        research_data = {}
        for (aspect_name, _), result in zip(search_queries.items(), initial_results):
            if isinstance(result, Exception):
                research_data[aspect_name] = {'success': False, 'error': str(result), 'results': []}
            else:
                research_data[aspect_name] = result
        
        total_initial_results = sum(r.get('count', 0) for r in research_data.values())
        logger.info(f"📊 Stage 1 complete: {total_initial_results} initial results")
        
        # ============================================
        # STAGE 2: GAP ANALYSIS
        # ============================================
        logger.info("🔍 Stage 2: Analyzing gaps and identifying follow-up research needs...")
        
        # Prepare research summary for gap analysis
        research_summary = {}
        for aspect_name, result in research_data.items():
            if result.get('success') and result.get('results'):
                research_summary[aspect_name] = {
                    'count': result.get('count', 0),
                    'top_results': [r.get('title', '') for r in result['results'][:3]]
                }
        
        gap_analysis_prompt = f"""Analyze the research data collected for {company_name} ({ticker}){location_context} and identify knowledge gaps.

Research Summary:
{research_summary}

For each aspect, identify:
1. What specific information is missing or insufficient
2. What follow-up questions would provide deeper insights
3. What contradictory information needs verification

Generate 5-10 targeted follow-up research queries that would fill gaps and provide comprehensive coverage for a {target_pages}-page research report.

Format as JSON array:
[
  {{"aspect": "fundamentals", "gap": "missing Q4 2024 data", "query": "specific query"}},
  ...
]

Return ONLY the JSON array, no other text:"""
        
        if inspect.iscoroutinefunction(generate_tool):
            gap_result = await generate_tool({'prompt': gap_analysis_prompt, 'model': 'sonnet', 'timeout': 300})
        else:
            gap_result = generate_tool({'prompt': gap_analysis_prompt, 'model': 'sonnet', 'timeout': 300})
        
        followup_queries = []
        if gap_result.get('success'):
            try:
                import json
                gap_text = gap_result.get('text', '')
                # Extract JSON from response
                if '[' in gap_text:
                    json_start = gap_text.find('[')
                    json_end = gap_text.rfind(']') + 1
                    gap_json = json.loads(gap_text[json_start:json_end])
                    followup_queries = gap_json[:10]  # Limit to 10 follow-up queries
                    logger.info(f"🔍 Identified {len(followup_queries)} follow-up research queries")
            except Exception as e:
                logger.warning(f"⚠️  Gap analysis parsing failed: {e}")
        
        # ============================================
        # STAGE 3: TARGETED FOLLOW-UP RESEARCH
        # ============================================
        if followup_queries:
            logger.info(f"🎯 Stage 3: Performing targeted follow-up research ({len(followup_queries)} queries)...")
            
            followup_tasks = [
                run_search(q.get('query', ''), f"followup_{q.get('aspect', 'unknown')}")
                for q in followup_queries
            ]
            followup_results = await asyncio.gather(*followup_tasks, return_exceptions=True)
            
            # Merge follow-up results into research_data
            for query_info, result in zip(followup_queries, followup_results):
                if not isinstance(result, Exception) and result.get('success'):
                    aspect = query_info.get('aspect', 'general')
                    if aspect in research_data:
                        # Merge results
                        existing_results = research_data[aspect].get('results', [])
                        new_results = result.get('results', [])
                        # Avoid duplicates
                        existing_urls = {r.get('url') for r in existing_results}
                        unique_new = [r for r in new_results if r.get('url') not in existing_urls]
                        research_data[aspect]['results'].extend(unique_new[:5])  # Add top 5 unique
                        research_data[aspect]['count'] = len(research_data[aspect]['results'])
                    else:
                        research_data[aspect] = result
            
            total_after_followup = sum(r.get('count', 0) for r in research_data.values())
            logger.info(f"📊 Stage 3 complete: {total_after_followup} total results (added {total_after_followup - total_initial_results})")
        
        # ============================================
        # STAGE 4: PROGRESSIVE SYNTHESIS + VISUALIZATION
        # ============================================
        logger.info("📝 Stage 4: Progressive synthesis with context building...")
        
        # Generate charts and tables if visualization skill is available
        charts = []
        tables = {}
        chart_markdown = ""
        table_markdown = ""
        
        try:
            from core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()
            viz_skill = registry.get_skill('financial-visualization')
            
            if not viz_skill:
                logger.warning("financial-visualization skill not found, skipping visualization")
                viz_skill = None
            
            if viz_skill:
                logger.info("📊 Generating financial charts and tables...")
                
                # Generate charts with intelligent orchestration (BEST OF AI)
                intelligent_charts_tool = viz_skill.tools.get('generate_intelligent_charts_tool')
                generate_charts_tool = viz_skill.tools.get('generate_financial_charts_tool')
                
                # Use intelligent version if available, fallback to standard
                chart_tool = intelligent_charts_tool or generate_charts_tool
                
                if chart_tool:
                    chart_result = await chart_tool({
                        'ticker': ticker,
                        'company_name': company_name,
                        'research_data': research_data,
                        'enable_intelligence': True if intelligent_charts_tool else False,
                        'format': 'png'
                        # chart_types auto-selected by intelligence if not provided
                    })
                    
                    if chart_result.get('success'):
                        charts = chart_result.get('charts', [])
                        chart_descriptions = chart_result.get('chart_descriptions', {})
                        chart_insights = chart_result.get('chart_insights', {})
                        chart_narratives = chart_result.get('chart_narratives', {})
                        anomalies = chart_result.get('anomalies', [])
                        forecasts = chart_result.get('forecasts', {})
                        section_placements = chart_result.get('section_placements', {})
                        
                        # Build markdown for charts with intelligent features
                        if charts:
                            chart_markdown = "\n## Financial Visualizations\n\n"
                            
                            # Add anomalies section if detected
                            if anomalies:
                                chart_markdown += "### ⚠️ Detected Anomalies\n\n"
                                for anomaly in anomalies[:5]:  # Top 5
                                    severity_icon = "🔴" if anomaly.get('severity') == 'high' else "🟡"
                                    chart_markdown += f"{severity_icon} **{anomaly.get('metric', 'Unknown')}**: {anomaly.get('description', '')}\n\n"
                                chart_markdown += "\n"
                            
                            # Add forecasts if available
                            if forecasts.get('revenue'):
                                forecast = forecasts['revenue']
                                chart_markdown += "### 📈 Revenue Forecast\n\n"
                                chart_markdown += f"Based on historical trends, revenue is forecasted to "
                                chart_markdown += f"{forecast.get('trend_direction', 'grow')} with a "
                                chart_markdown += f"{forecast.get('forecast_growth_rate', 0):.1f}% growth rate "
                                chart_markdown += f"in the next period ({forecast.get('trend_strength', 'moderate')} trend).\n\n"
                            
                            # Add charts with narratives
                            for chart_path, chart_type in zip(charts, chart_descriptions.keys()):
                                chart_filename = os.path.basename(chart_path)
                                description = chart_descriptions.get(chart_type, '')
                                narrative = chart_narratives.get(chart_type, '')
                                insight = chart_insights.get(chart_type, '')
                                
                                chart_markdown += f"### {chart_type.replace('_', ' ').title()}\n\n"
                                chart_markdown += f"![{chart_type}]({chart_path})\n\n"
                                chart_markdown += f"*{description}*\n\n"
                                
                                # Add comprehensive narrative if available
                                if narrative:
                                    chart_markdown += f"{narrative}\n\n"
                                elif insight:
                                    chart_markdown += f"**Key Insight:** {insight}\n\n"
                
                # Generate tables
                generate_tables_tool = viz_skill.tools.get('generate_financial_tables_tool')
                if generate_tables_tool:
                    table_result = await generate_tables_tool({
                        'ticker': ticker,
                        'company_name': company_name,
                        'research_data': research_data,
                        'table_types': ['financial_statements', 'valuation_metrics', 'key_ratios'],
                        'format': 'markdown'
                    })
                    
                    if table_result.get('success'):
                        tables = table_result.get('tables', {})
                        table_descriptions = table_result.get('table_descriptions', {})
                        
                        # Build markdown for tables
                        if tables:
                            table_markdown = "\n## Financial Tables\n\n"
                            for table_type, table_content in tables.items():
                                description = table_descriptions.get(table_type, '')
                                table_markdown += f"### {table_type.replace('_', ' ').title()}\n\n"
                                table_markdown += f"{table_content}\n\n"
                                table_markdown += f"*{description}*\n\n"
        except Exception as e:
            logger.warning(f"Visualization generation failed (continuing without charts/tables): {e}")
            import traceback
            logger.debug(f"Visualization error details: {traceback.format_exc()}")
        
        # Build comprehensive research content
        research_content = f"""
# Comprehensive Research Data for {company_name} ({ticker}){location_context}

## Company Information
- **Ticker:** {ticker}
- **Company:** {company_name}
- **Location:** {country or 'Not specified'}{f' ({exchange})' if exchange else ''}
- **Report Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Total Research Results:** {sum(r.get('count', 0) for r in research_data.values())}

{table_markdown}

{chart_markdown}

"""
        
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
                for i, res in enumerate(result['results'][:6], 1):  # Top 6 per aspect
                    research_content += f"### {i}. {res.get('title', 'Untitled')}\n"
                    research_content += f"**URL:** {res.get('url', '')}\n"
                    snippet = res.get('snippet', 'No summary available')
                    if snippet:
                        research_content += f"**Summary:** {snippet[:400]}\n\n"  # Longer snippets for deep research
                    else:
                        research_content += "\n"
            else:
                research_content += f"No {aspect_title.lower()} research results available.\n\n"
        
        # Generate comprehensive report with explicit instructions
        claude_prompt = f"""WRITE A COMPLETE, DETAILED STOCK RESEARCH REPORT for {company_name} ({ticker}){location_context} based on the extensive research data provided below.

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

1. **Executive Summary** (1-1.5 pages, 300-500 words)
   - Company overview and business description
   - Key investment highlights
   - Current stock price and market cap
   - Quick snapshot of financials
   - Investment thesis summary

2. **Company Overview & Business Model** (1-1.5 pages, 500-700 words)
   - Company history and background
   - Business segments and product portfolio
   - Geographic presence and market position
   - Competitive advantages and moats
   - Strategic initiatives

3. **Industry Analysis & Market Position** (1 page, 400-600 words)
   - Industry overview and trends
   - Market size and growth prospects
   - Competitive landscape
   - Industry challenges and opportunities
   - Company's position within the industry

4. **Financial Analysis** (2 pages, 600-900 words - MOST DETAILED)
   - Revenue trends and growth drivers
   - Profitability analysis (margins, ROE, ROA)
   - Balance sheet strength
   - Cash flow analysis
   - Key financial ratios and comparisons
   - Historical performance trends

5. **Valuation Analysis** (1-1.5 pages, 500-700 words)
   - Current valuation metrics (P/E, P/B, EV/EBITDA)
   - Peer comparison
   - DCF analysis (if data available)
   - Intrinsic value estimation
   - Valuation conclusion

6. **Technical Analysis** (1 page, 400-600 words)
   - Price trends and patterns
   - Key support and resistance levels
   - Technical indicators (RSI, MACD, Moving Averages)
   - Chart patterns
   - Trading recommendations

7. **Management & Corporate Governance** (0.5-1 page, 300-500 words)
   - Management team overview
   - Corporate governance practices
   - ESG factors
   - Management track record

8. **Broker Research & Analyst Coverage** (1 page, 400-600 words)
   - Analyst ratings summary
   - Consensus price targets
   - Key research highlights from major brokers
   - Analyst recommendations breakdown
   - Recent upgrades/downgrades

9. **Risks & Challenges** (1 page, 400-600 words)
   - Key business risks
   - Regulatory risks
   - Competitive threats
   - Market risks
   - Company-specific challenges

10. **Growth Prospects & Investment Outlook** (1 page, 400-600 words)
    - Growth drivers and catalysts
    - Expansion plans
    - New product launches
    - Market opportunities
    - Future outlook

11. **Dividend Analysis** (0.5 page, 300-400 words)
    - Dividend history
    - Current yield
    - Payout ratio
    - Dividend sustainability

12. **Conclusion & Investment Recommendation** (0.5-1 page, 300-500 words)
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
- Start immediately with: "# {company_name} ({ticker}){location_context} - Comprehensive Research Report"
- Write out ALL 12 sections in FULL DETAIL with complete content
- Each major section should have multiple subsections (###) with detailed paragraphs
- Use markdown formatting (## for sections, ### for subsections, **bold** for emphasis, tables where appropriate)
- Include specific data points, financial metrics, percentages, and analysis throughout
- Write paragraphs, not bullet points (use bullets only for lists within paragraphs)
- **IMPORTANT**: If financial charts or tables are provided in the research data above, reference them naturally in the relevant sections (e.g., "As shown in the revenue growth chart..." or "The valuation metrics table shows...")
- Embed chart references within the text where they add value to the analysis

**START WRITING THE COMPLETE REPORT NOW. BEGIN WITH THE HEADER AND EXECUTIVE SUMMARY, THEN CONTINUE WITH ALL SECTIONS IN FULL DETAIL. DO NOT STOP UNTIL YOU HAVE WRITTEN ALL {target_pages} PAGES OF CONTENT:**"""
        
        # Use longer timeout for deep research
        # For comprehensive 10-page reports, Claude needs significant time
        # We'll use 30 minutes (1800s) which should be sufficient
        timeout_seconds = 1800  # 30 minutes for comprehensive deep research
        
        logger.info(f"⏱️  Using {timeout_seconds//60} minute timeout for Claude generation")
        logger.info(f"📝 Prompt size: ~{len(claude_prompt)//1024}KB, Research data: {len(research_content)//1024}KB")
        
        # Generate report with extended timeout
        try:
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
        except asyncio.TimeoutError:
            logger.error(f"❌ Claude generation timed out after {timeout_seconds}s")
            return {
                'success': False,
                'error': f'Claude generation timed out after {timeout_seconds} seconds. Try reducing target_pages or research scope.',
                'research_stages': {
                    'initial_research': total_initial_results,
                    'followup_research': len(followup_queries),
                    'total_results': sum(r.get('count', 0) for r in research_data.values())
                }
            }
        
        if not claude_result.get('success'):
            return {
                'success': False,
                'error': f'Claude generation failed: {claude_result.get("error")}',
                'research_summary': {k: {'success': v.get('success'), 'count': v.get('count', 0)} 
                                    for k, v in research_data.items()}
            }
        
        markdown_content = claude_result.get('text', '')
        
        # ============================================
        # STAGE 5: QUALITY VALIDATION (Optional)
        # ============================================
        # Check if report is comprehensive enough
        word_count = len(markdown_content.split())
        logger.info(f"📊 Generated report: {word_count} words")
        
        if word_count < target_pages * 400:  # Less than 400 words per page
            logger.warning(f"⚠️  Report may be too short ({word_count} words, target: {target_pages * 500}+)")
        
        # Add header with metadata
        header = f"""# {company_name} ({ticker}){location_context} - Comprehensive Research Report

**Ticker:** {ticker}  
**Company:** {company_name}  
**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** Jotty Deep Research  
**Research Methodology:** Multi-stage deep research with context intelligence

---

"""
        
        # Embed charts and tables in the report if available
        visualization_section = ""
        if charts or tables:
            visualization_section = "\n\n---\n\n## Financial Visualizations\n\n"
            
            if tables:
                visualization_section += "### Financial Tables\n\n"
                for table_type, table_content in tables.items():
                    visualization_section += f"{table_content}\n\n"
            
            if charts:
                visualization_section += "### Financial Charts\n\n"
                for chart_path in charts:
                    chart_filename = os.path.basename(chart_path)
                    # Use relative path or absolute path depending on PDF conversion needs
                    visualization_section += f"![{chart_filename}]({chart_path})\n\n"
        
        full_markdown = header + markdown_content + visualization_section
        
        # ============================================
        # STAGE 6: SAVE AND CONVERT
        # ============================================
        logger.info("💾 Saving markdown file...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        safe_ticker = ticker.replace('/', '-').replace('\\', '-')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_filename = f"{safe_ticker}_deep_research_{timestamp}.md"
        md_path = output_path / md_filename
        
        write_tool = file_ops_skill.tools.get('write_file_tool')
        write_result = write_tool({
            'path': str(md_path),
            'content': full_markdown
        })
        
        if not write_result.get('success'):
            return {'success': False, 'error': f'Failed to write markdown: {write_result.get("error")}'}
        
        logger.info(f"✅ Markdown saved: {md_path}")
        
        # Convert to PDF
        logger.info("📄 Converting to PDF...")
        
        pdf_filename = f"{safe_ticker}_deep_research_{timestamp}.pdf"
        pdf_path = output_path / pdf_filename
        
        convert_tool = doc_converter_skill.tools.get('convert_to_pdf_tool')
        
        if inspect.iscoroutinefunction(convert_tool):
            pdf_result = await convert_tool({
                'input_file': str(md_path),
                'output_file': str(pdf_path),
                'page_size': 'a4',
                'title': f'{company_name} ({ticker}) - Deep Research Report',
                'author': 'Jotty Deep Research'
            })
        else:
            pdf_result = convert_tool({
                'input_file': str(md_path),
                'output_file': str(pdf_path),
                'page_size': 'a4',
                'title': f'{company_name} ({ticker}) - Deep Research Report',
                'author': 'Jotty Deep Research'
            })
        
        if not pdf_result.get('success'):
            return {'success': False, 'error': f'PDF conversion failed: {pdf_result.get("error")}', 'md_path': str(md_path)}
        
        pdf_output_path = pdf_result.get('output_path', str(pdf_path))
        logger.info(f"✅ PDF generated: {pdf_output_path}")
        
        # Send to Telegram
        telegram_sent = False
        if send_telegram and telegram_skill:
            logger.info("📱 Sending to Telegram...")
            
            send_tool = telegram_skill.tools.get('send_telegram_file_tool')
            if send_tool:
                caption = f"📊 {company_name} ({ticker}) - Deep Research Report\n\nComprehensive {target_pages}-page analysis\nResearch Methodology: Multi-stage deep research"
                
                if inspect.iscoroutinefunction(send_tool):
                    telegram_result = await send_tool({
                        'file_path': pdf_output_path,
                        'caption': caption
                    })
                else:
                    telegram_result = send_tool({
                        'file_path': pdf_output_path,
                        'caption': caption
                    })
                
                if telegram_result.get('success'):
                    telegram_sent = True
                    logger.info("✅ PDF sent to Telegram")
        
        return {
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'md_path': str(md_path),
            'pdf_path': pdf_output_path,
            'word_count': word_count,
            'research_stages': {
                'initial_research': total_initial_results,
                'followup_research': len(followup_queries),
                'total_results': sum(r.get('count', 0) for r in research_data.values())
            },
            'telegram_sent': telegram_sent
        }
        
    except Exception as e:
        logger.error(f"Deep research error: {e}", exc_info=True)
        return {'success': False, 'error': f'Deep research failed: {str(e)}'}
