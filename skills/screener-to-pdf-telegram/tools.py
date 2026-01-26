"""
Screener.in → Analysis → PDF → Telegram Pipeline Skill

Complete workflow:
1. Fetch financial data from screener.in
2. Synthesize with Claude CLI LLM
3. Generate PDF
4. Send to Telegram
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


async def screener_analyze_pdf_telegram_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete workflow: Screener.in → Analysis → PDF → Telegram
    
    Args:
        params: Dictionary containing:
            - symbols (str or list, required): Company symbol(s) - e.g., "RELIANCE" or ["RELIANCE", "TCS"]
            - analysis_type (str, optional): 'comprehensive', 'quick', 'ratios_only', default: 'comprehensive'
            - telegram_chat_id (str, optional): Telegram chat ID
            - telegram_token (str, optional): Telegram bot token
            - output_dir (str, optional): Output directory, default: './output'
            - title (str, optional): Custom PDF title
            - use_proxy (bool, optional): Use proxy for screener.in, default: True
    
    Returns:
        Dictionary with workflow results
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        symbols = params.get('symbols')
        if not symbols:
            return {
                'success': False,
                'error': 'symbols parameter is required'
            }
        
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        analysis_type = params.get('analysis_type', 'comprehensive')
        output_dir = Path(params.get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        registry = get_skills_registry()
        registry.init()
        
        logger.info(f"📊 Screener.in → Analysis → PDF → Telegram workflow: {symbols}")
        
        # Step 1: Fetch financial data from screener.in
        logger.info("📡 Step 1: Fetching financial data from screener.in...")
        
        screener_skill = registry.get_skill('screener-financials')
        if not screener_skill:
            return {
                'success': False,
                'error': 'screener-financials skill not available'
            }
        
        get_financials_tool = screener_skill.tools.get('get_company_financials_tool')
        if not get_financials_tool:
            return {
                'success': False,
                'error': 'get_company_financials_tool not found'
            }
        
        companies_data = []
        companies_analyzed = []
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            logger.info(f"   Fetching data for {symbol}...")
            
            financials_result = get_financials_tool({
                'company_name': symbol,
                'data_type': 'all',
                'period': 'annual',
                'format': 'json',
                'use_proxy': params.get('use_proxy', True),
                'max_retries': 3
            })
            
            if financials_result.get('success'):
                companies_data.append(financials_result)
                companies_analyzed.append({
                    'symbol': symbol,
                    'name': financials_result.get('company_name', symbol),
                    'code': financials_result.get('company_code', symbol)
                })
                logger.info(f"   ✅ Fetched data for {symbol}")
            else:
                logger.warning(f"   ⚠️  Failed to fetch {symbol}: {financials_result.get('error')}")
        
        if not companies_data:
            return {
                'success': False,
                'error': 'Failed to fetch data for any company'
            }
        
        # Step 2: Prepare data for analysis
        logger.info("📝 Step 2: Preparing data for analysis...")
        
        analysis_prompt = _build_analysis_prompt(companies_data, analysis_type)
        
        # Step 3: Synthesize with Claude CLI LLM
        logger.info("🤖 Step 3: Analyzing with Claude LLM...")
        
        claude_skill = registry.get_skill('claude-cli-llm')
        if not claude_skill:
            return {
                'success': False,
                'error': 'claude-cli-llm skill not available'
            }
        
        summarize_tool = claude_skill.tools.get('summarize_text_tool')
        if not summarize_tool:
            # Try alternative tool name
            summarize_tool = claude_skill.tools.get('analyze_text_tool')
        
        if not summarize_tool:
            return {
                'success': False,
                'error': 'Claude LLM summarize tool not found'
            }
        
        # Format financial data as text for analysis
        financial_text = _format_financial_data_for_analysis(companies_data)
        
        # Build full prompt with data
        full_prompt = f"{analysis_prompt}\n\n{financial_text}"
        
        analysis_result = summarize_tool({
            'content': full_prompt,
            'model': params.get('model', 'sonnet'),
            'max_tokens': params.get('max_tokens', 4000)
        })
        
        if not analysis_result.get('success'):
            return {
                'success': False,
                'error': f'Analysis failed: {analysis_result.get("error")}'
            }
        
        analysis_text = analysis_result.get('summary') or analysis_result.get('output', '')
        
        # Step 4: Create markdown document
        logger.info("📄 Step 4: Creating markdown document...")
        
        markdown_content = _create_markdown_report(companies_analyzed, companies_data, analysis_text, analysis_type)
        
        # Save markdown temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_str = '_'.join([c['code'] for c in companies_analyzed])
        markdown_file = output_dir / f'{symbol_str}_analysis_{timestamp}.md'
        markdown_file.write_text(markdown_content, encoding='utf-8')
        
        # Step 5: Convert to PDF
        logger.info("📑 Step 5: Converting to PDF...")
        
        document_converter_skill = registry.get_skill('document-converter')
        if not document_converter_skill:
            return {
                'success': False,
                'error': 'document-converter skill not available'
            }
        
        convert_pdf_tool = document_converter_skill.tools.get('convert_to_pdf_tool')
        if not convert_pdf_tool:
            return {
                'success': False,
                'error': 'convert_to_pdf_tool not found'
            }
        
        pdf_title = params.get('title') or f"Financial Analysis: {', '.join([c['name'] for c in companies_analyzed])}"
        
        pdf_output_file = output_dir / f'{symbol_str}_analysis_{timestamp}.pdf'
        
        pdf_result = convert_pdf_tool({
            'input_file': str(markdown_file),
            'output_file': str(pdf_output_file),
            'title': pdf_title,
            'author': 'Jotty Financial Analysis',
            'page_size': 'a4'
        })
        
        if not pdf_result.get('success'):
            return {
                'success': False,
                'error': f'PDF conversion failed: {pdf_result.get("error")}'
            }
        
        # Try multiple possible return keys
        pdf_path = (
            pdf_result.get('output_path') or  # document-converter returns 'output_path'
            pdf_result.get('output_file') or 
            pdf_result.get('pdf_path')
        )
        
        # Verify PDF was created
        if not pdf_path or not Path(pdf_path).exists():
            # Try the expected output file
            if pdf_output_file.exists():
                pdf_path = str(pdf_output_file)
            else:
                logger.warning(f"PDF file not found at expected location, checking output_dir")
                # Look for PDF files in output_dir matching our pattern
                pdf_files = list(output_dir.glob(f'{symbol_str}_analysis_*.pdf'))
                if pdf_files:
                    pdf_path = str(pdf_files[-1])  # Use most recent
                    logger.info(f"Found PDF: {pdf_path}")
                else:
                    logger.error(f"PDF file not created at {pdf_output_file}")
                    return {
                        'success': False,
                        'error': f'PDF file was not created at expected location: {pdf_output_file}'
                    }
        
        # Step 6: Send to Telegram
        telegram_sent = False
        telegram_chat_id = params.get('telegram_chat_id')
        
        if telegram_chat_id:
            logger.info("📱 Step 6: Sending to Telegram...")
            
            telegram_skill = registry.get_skill('telegram-sender')
            if telegram_skill:
                send_file_tool = telegram_skill.tools.get('send_telegram_file_tool')
                if send_file_tool:
                    telegram_result = send_file_tool({
                        'file_path': pdf_path,
                        'chat_id': telegram_chat_id,
                        'token': params.get('telegram_token'),
                        'caption': f"📊 Financial Analysis: {', '.join([c['name'] for c in companies_analyzed])}"
                    })
                    
                    if telegram_result.get('success'):
                        telegram_sent = True
                        logger.info("   ✅ Sent to Telegram")
                    else:
                        logger.warning(f"   ⚠️  Telegram send failed: {telegram_result.get('error')}")
        
        return {
            'success': True,
            'companies_analyzed': companies_analyzed,
            'pdf_path': pdf_path,
            'markdown_path': str(markdown_file),
            'telegram_sent': telegram_sent,
            'analysis': analysis_text,
            'symbols': symbols
        }
        
    except Exception as e:
        logger.error(f"Screener.in → PDF → Telegram workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


def _build_analysis_prompt(companies_data: List[Dict[str, Any]], analysis_type: str) -> str:
    """Build analysis prompt based on analysis type."""
    
    if analysis_type == 'ratios_only':
        return """Analyze the key financial ratios for these companies. Focus on:
- Profitability ratios (ROE, ROCE, P/E, P/B)
- Market valuation metrics
- Dividend yield
- Growth trends

Provide a concise comparison and investment insights."""
    
    elif analysis_type == 'quick':
        return """Provide a quick financial analysis focusing on:
- Key strengths and weaknesses
- Main financial metrics
- Investment recommendation (Buy/Hold/Sell)

Keep it concise (2-3 paragraphs)."""
    
    else:  # comprehensive
        return """Provide a comprehensive financial analysis covering:

1. **Company Overview**: Business model and industry position
2. **Financial Performance**: 
   - Revenue and profit trends
   - Profitability ratios (ROE, ROCE, margins)
   - Growth rates (revenue, profit, stock price)
3. **Financial Health**:
   - Balance sheet strength
   - Debt levels and liquidity
   - Cash flow analysis
4. **Valuation**:
   - P/E, P/B ratios
   - Market cap analysis
   - Dividend yield
5. **Investment Thesis**:
   - Key strengths
   - Key risks
   - Investment recommendation with reasoning

Be thorough but clear. Use data from the financial statements provided."""


def _format_financial_data_for_analysis(companies_data: List[Dict[str, Any]]) -> str:
    """Format financial data as text for LLM analysis."""
    lines = []
    
    for company_data in companies_data:
        company_name = company_data.get('company_name', 'Unknown')
        company_code = company_data.get('company_code', 'Unknown')
        data = company_data.get('data', {})
        
        lines.append(f"\n{'='*60}")
        lines.append(f"COMPANY: {company_name} ({company_code})")
        lines.append(f"{'='*60}\n")
        
        # Ratios
        ratios = data.get('ratios', {})
        if ratios:
            lines.append("KEY FINANCIAL RATIOS:")
            for name, value in sorted(ratios.items()):
                lines.append(f"  - {name}: {value}")
            lines.append("")
        
        # P&L Summary
        pl_data = data.get('profit_loss', {})
        if pl_data and pl_data.get('rows'):
            lines.append("PROFIT & LOSS SUMMARY:")
            headers = pl_data.get('headers', [])
            rows = pl_data.get('rows', [])
            
            # Show key metrics
            key_metrics = ['Sales+', 'Operating Profit', 'Net Profit', 'EBITDA']
            for row in rows[:10]:  # First 10 rows usually contain key metrics
                if row and len(row) > 0:
                    metric_name = row[0]
                    if any(key in metric_name for key in key_metrics):
                        values = row[1:min(6, len(row))]  # Show first 5 periods
                        lines.append(f"  {metric_name}: {', '.join(values)}")
            lines.append("")
        
        # Balance Sheet Summary
        bs_data = data.get('balance_sheet', {})
        if bs_data and bs_data.get('rows'):
            lines.append("BALANCE SHEET SUMMARY:")
            rows = bs_data.get('rows', [])
            key_items = ['Equity Capital', 'Reserves', 'Borrowings', 'Total Assets']
            for row in rows[:8]:
                if row and len(row) > 0:
                    item_name = row[0]
                    if any(key in item_name for key in key_items):
                        values = row[1:min(6, len(row))]
                        lines.append(f"  {item_name}: {', '.join(values)}")
            lines.append("")
    
    return "\n".join(lines)


def _create_markdown_report(
    companies_analyzed: List[Dict[str, Any]],
    companies_data: List[Dict[str, Any]],
    analysis_text: str,
    analysis_type: str
) -> str:
    """Create markdown report from analysis."""
    lines = []
    
    # Title
    company_names = ', '.join([c['name'] for c in companies_analyzed])
    lines.append(f"# Financial Analysis: {company_names}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Analysis Type:** {analysis_type.title()}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(analysis_text)
    lines.append("")
    
    # Detailed Financial Data
    lines.append("## Detailed Financial Data")
    lines.append("")
    
    for i, (company_info, company_data) in enumerate(zip(companies_analyzed, companies_data), 1):
        lines.append(f"### {i}. {company_info['name']} ({company_info['code']})")
        lines.append("")
        
        # Ratios
        ratios = company_data.get('data', {}).get('ratios', {})
        if ratios:
            lines.append("#### Key Ratios")
            lines.append("")
            lines.append("| Ratio | Value |")
            lines.append("|-------|-------|")
            for name, value in sorted(ratios.items()):
                lines.append(f"| {name} | {value} |")
            lines.append("")
        
        # P&L Table
        pl_data = company_data.get('data', {}).get('profit_loss', {})
        if pl_data and pl_data.get('rows'):
            lines.append("#### Profit & Loss Statement")
            lines.append("")
            headers = pl_data.get('headers', [])
            rows = pl_data.get('rows', [])
            
            if headers:
                lines.append("| " + " | ".join(headers[:6]) + " |")
                lines.append("|" + "|".join(["---"] * min(6, len(headers))) + "|")
                
                for row in rows[:15]:  # Show first 15 rows
                    if row and len(row) > 0:
                        cells = row[:min(6, len(row))]
                        lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    
    return "\n".join(lines)
