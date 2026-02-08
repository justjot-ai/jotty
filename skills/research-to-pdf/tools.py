"""
research-to-pdf: Research a topic and generate PDF report using Jotty skills.

This skill combines:
- last30days-claude-cli: Research topics from last 30 days
- document-converter: Convert markdown to PDF
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Module-level status callback (set by executor before calling tool)
_current_status_callback: Optional[Callable] = None


def emit_status(stage: str, detail: str = ""):
    """Emit a status update if callback is set."""
    if _current_status_callback:
        try:
            _current_status_callback(stage, detail)
        except Exception:
            pass
    logger.info(f"[research-to-pdf] {stage}: {detail}")


async def research_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research a topic and generate a PDF report.
    
    Args:
        params: Dictionary containing:
            - topic (required): Topic to research
            - output_dir (optional): Output directory (default: ~/jotty/reports)
            - title (optional): Report title (default: auto-generated from topic)
            - author (optional): Report author (default: 'Jotty Framework')
            - page_size (optional): PDF page size - 'a4', 'a5', 'a6', 'letter' (default: 'a4')
            - deep (optional): Deep research mode (default: False)
            - quick (optional): Quick research mode (default: False)
    
    Returns:
        Dict with:
            - success (bool): Whether operation succeeded
            - pdf_path (str): Path to generated PDF
            - md_path (str): Path to markdown file
            - file_size (int): PDF file size in bytes
            - error (str, optional): Error message if failed
    """
    global _current_status_callback

    try:
        # Extract status callback from params (passed by executor)
        _current_status_callback = params.pop('_status_callback', None)

        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry

        topic = params.get('topic', '')
        if not topic:
            return {
                'success': False,
                'error': 'topic parameter is required'
            }
        
        registry = get_skills_registry()
        registry.init()

        # Step 1: Research
        emit_status("Researching", f"🔍 Searching for {topic}...")
        last30days_skill = registry.get_skill('last30days-claude-cli')
        if not last30days_skill:
            return {
                'success': False,
                'error': 'last30days-claude-cli skill not found'
            }
        
        research_tool = last30days_skill.tools.get('last30days_claude_cli_tool')
        if not research_tool:
            return {
                'success': False,
                'error': 'last30days_claude_cli_tool not found'
            }
        
        # Determine research depth
        deep = params.get('deep', False)
        quick = params.get('quick', False)
        
        research_result = await research_tool({
            'topic': topic,
            'deep': deep,
            'quick': quick,
            'emit': 'md'  # Get markdown format
        })

        if not research_result.get('success'):
            return {
                'success': False,
                'error': f'Research failed: {research_result.get("error")}'
            }

        raw_research = research_result.get('output', '')
        emit_status("Analyzing", "🧠 Analyzing search results...")

        # Step 1.5: AI Synthesis - Convert raw search results into comprehensive analysis
        research_content = await _synthesize_research(topic, raw_research, deep)
        
        # Step 2: Create markdown file
        # Default to stock_market/outputs directory
        # __file__ is skills/research-to-pdf/tools.py
        # Go up: research-to-pdf -> skills -> Jotty -> stock_market
        current_file = Path(__file__).resolve()
        # From skills/research-to-pdf/tools.py: up 3 levels to Jotty, then up 1 to stock_market
        jotty_dir = current_file.parent.parent.parent
        stock_market_root = jotty_dir.parent
        default_output = stock_market_root / 'outputs'
        
        # Ensure directory exists or use fallback
        if not default_output.exists():
            default_output.mkdir(parents=True, exist_ok=True)
        
        # Use provided output_dir or default
        output_dir = Path(params.get('output_dir', str(default_output)))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_topic = topic.replace(' ', '_').replace('/', '_')[:50]  # Sanitize filename
        md_file = output_dir / f'{safe_topic}_research_{timestamp}.md'
        
        # Generate report title
        report_title = params.get('title', f'{topic.title()} Research Report')
        author = params.get('author', 'Jotty Framework')
        
        # Format research content with proper wrapping
        formatted_content = _format_markdown_for_pdf(research_content)
        
        # Create comprehensive markdown report
        report_content = f"""# {report_title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Research Period:** Last 30 days
**Source:** Jotty last30days-claude-cli skill
**Author:** {author}

---

{formatted_content}

---

## Report Metadata

- **Research Tool:** last30days-claude-cli (Jotty)
- **Search Engine:** DuckDuckGo via ddgs library
- **Generated by:** Jotty Framework
- **Format:** Markdown → PDF
- **Topic:** {topic}
"""
        
        emit_status("Writing", "📝 Creating markdown report...")
        md_file.write_text(report_content)

        # Step 3: Convert to PDF
        emit_status("Converting", "📄 Converting to PDF...")
        doc_converter_skill = registry.get_skill('document-converter')
        if not doc_converter_skill:
            return {
                'success': False,
                'error': 'document-converter skill not found'
            }
        
        pdf_tool = doc_converter_skill.tools.get('convert_to_pdf_tool')
        if not pdf_tool:
            return {
                'success': False,
                'error': 'convert_to_pdf_tool not found'
            }
        
        page_size = params.get('page_size', 'a4')
        pdf_result = pdf_tool({
            'input_file': str(md_file),
            'output_file': str(md_file.with_suffix('.pdf')),
            'title': report_title,
            'author': author,
            'page_size': page_size
        })
        
        if not pdf_result.get('success'):
            return {
                'success': False,
                'error': f'PDF conversion failed: {pdf_result.get("error")}',
                'md_path': str(md_file)
            }
        
        pdf_path = pdf_result.get('output_path')
        file_size = pdf_result.get('file_size', 0)
        
        return {
            'success': True,
            'pdf_path': pdf_path,
            'md_path': str(md_file),
            'file_size': file_size,
            'topic': topic,
            'title': report_title
        }
        
    except Exception as e:
        logger.error(f"Error in research_to_pdf_tool: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

async def _synthesize_research(topic: str, raw_research: str, deep: bool = False) -> str:
    """
    Use AI to synthesize raw search results into comprehensive analysis.

    Instead of just dumping links, this creates a proper research report
    with analysis, insights, and structured sections.
    """
    try:
        # Try to use claude-cli-llm skill for synthesis
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        claude_skill = registry.get_skill('claude-cli-llm')

        if claude_skill:
            llm_tool = claude_skill.tools.get('claude_cli_llm_tool')
            if llm_tool:
                # Determine target length based on depth
                target_pages = 15 if deep else 8
                target_words = target_pages * 500  # ~500 words per page

                synthesis_prompt = f"""You are a senior research analyst. Based on the following research data about "{topic}", write a comprehensive, well-structured research report.

## Requirements:
- Write approximately {target_words} words ({target_pages} pages)
- Use proper markdown formatting with headers, subheaders, bullet points
- Include executive summary, key findings, detailed analysis, and conclusions
- Cite specific facts, statistics, and data points from the research
- Provide balanced analysis covering multiple perspectives
- Include actionable insights and recommendations
- DO NOT just list links - synthesize and analyze the information

## Report Structure:
1. **Executive Summary** (1 page) - Key highlights and main findings
2. **Background & Context** (1-2 pages) - Industry context, history, relevance
3. **Detailed Analysis** (3-5 pages) - In-depth analysis of key aspects
4. **Comparative Analysis** (1-2 pages) - If comparing entities, detailed comparison
5. **Key Insights & Trends** (1 page) - Important patterns and observations
6. **Risks & Challenges** (0.5-1 page) - Potential issues and concerns
7. **Future Outlook** (0.5-1 page) - Predictions and expectations
8. **Conclusion & Recommendations** (0.5 page) - Summary and actionable advice

## Raw Research Data:
{raw_research[:15000]}

Write the comprehensive research report now:"""

                import inspect
                if inspect.iscoroutinefunction(llm_tool):
                    result = await llm_tool({'prompt': synthesis_prompt, 'max_tokens': 8000})
                else:
                    result = llm_tool({'prompt': synthesis_prompt, 'max_tokens': 8000})

                if result.get('success') and result.get('response'):
                    logger.info(f"AI synthesis complete: {len(result['response'])} chars")
                    return result['response']

        # Fallback: Try DSPy
        import dspy
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
            class ResearchSynthesis(dspy.Signature):
                """Synthesize raw research into comprehensive analysis report."""
                topic: str = dspy.InputField()
                raw_research: str = dspy.InputField()
                target_pages: int = dspy.InputField()
                report: str = dspy.OutputField(desc="Comprehensive markdown research report with sections, analysis, insights")

            predictor = dspy.Predict(ResearchSynthesis)
            target_pages = 15 if deep else 8
            result = predictor(topic=topic, raw_research=raw_research[:10000], target_pages=target_pages)
            if result.report:
                logger.info(f"DSPy synthesis complete: {len(result.report)} chars")
                return result.report

    except Exception as e:
        logger.warning(f"AI synthesis failed, using raw research: {e}")

    # Fallback: Return formatted raw research
    return raw_research


def _format_markdown_for_pdf(content: str) -> str:
    """
    Format markdown content for better PDF rendering.
    
    Handles:
    - Long lines wrapping (Pandoc handles this, but we format URLs and lists better)
    - URL formatting with proper line breaks
    - List formatting
    - Code blocks
    - Better paragraph separation
    """
    import textwrap
    
    lines = content.split('\n')
    formatted_lines = []
    in_code_block = False
    in_list = False
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # Track code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            formatted_lines.append(line)
            continue
        
        # Inside code blocks - preserve exactly
        if in_code_block:
            formatted_lines.append(line)
            continue
        
        # Empty lines - preserve
        if not stripped:
            formatted_lines.append('')
            in_list = False
            continue
        
        # Headers - keep as is
        if stripped.startswith('#'):
            formatted_lines.append(line)
            in_list = False
            continue
        
        # Horizontal rules
        if stripped.startswith('---') or stripped.startswith('***'):
            formatted_lines.append(line)
            in_list = False
            continue
        
        # Lists - format better
        if stripped.startswith(('-', '*', '+', '1.', '2.', '3.', '4.', '5.')):
            in_list = True
            # Extract list marker and content
            marker_match = None
            for marker in ['- ', '* ', '+ ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                if stripped.startswith(marker):
                    marker_match = marker
                    break
            
            if marker_match:
                content_part = stripped[len(marker_match):].strip()
                # Wrap long list items
                if len(content_part) > 75:
                    wrapped = textwrap.wrap(content_part, width=75, 
                                           initial_indent='  ', 
                                           subsequent_indent='  ')
                    formatted_lines.append(f"{marker_match}{wrapped[0]}")
                    for wrap_line in wrapped[1:]:
                        formatted_lines.append(wrap_line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
            continue
        
        # URLs on their own line - format better
        if stripped.startswith('http://') or stripped.startswith('https://'):
            # Long URLs - break them if needed
            if len(stripped) > 100:
                # Try to break at path segments
                url_parts = stripped.split('/')
                if len(url_parts) > 3:
                    base = '/'.join(url_parts[:3])
                    path = '/'.join(url_parts[3:])
                    formatted_lines.append(base)
                    formatted_lines.append('  ' + path)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
            continue
        
        # Lines with URLs in them - format better
        if 'http' in line:
            # Handle **URL:** format
            if '**URL:**' in line:
                parts = line.split('**URL:**')
                if len(parts) == 2:
                    formatted_lines.append(parts[0].rstrip() + ' **URL:**')
                    url_part = parts[1].strip()
                    # Break long URLs at path segments or wrap
                    if len(url_part) > 85:
                        # Try to break at / characters
                        if url_part.count('/') > 3:
                            url_parts = url_part.split('/', 3)
                            base = '/'.join(url_parts[:3]) + '/'
                            path = url_parts[3] if len(url_parts) > 3 else ''
                            formatted_lines.append('  ' + base)
                            if path:
                                # Wrap the path part
                                wrapped_path = textwrap.wrap(path, width=85,
                                                           initial_indent='  ',
                                                           subsequent_indent='  ')
                                formatted_lines.extend(wrapped_path)
                        else:
                            # Just wrap the URL
                            wrapped = textwrap.wrap(url_part, width=85,
                                                  initial_indent='  ',
                                                  subsequent_indent='  ',
                                                  break_long_words=False)
                            formatted_lines.extend(wrapped)
                    else:
                        formatted_lines.append('  ' + url_part)
                else:
                    formatted_lines.append(line)
                continue
            # Handle other URL patterns in text
            elif len(stripped) > 100:
                # Extract URLs and wrap text around them
                import re
                url_pattern = r'https?://[^\s\)]+'
                urls = re.findall(url_pattern, line)
                if urls:
                    # Replace URLs with placeholders, wrap text, then restore URLs
                    text_without_urls = line
                    for i, url in enumerate(urls):
                        text_without_urls = text_without_urls.replace(url, f'__URL_{i}__', 1)
                    
                    # Wrap the text
                    wrapped_text = textwrap.wrap(text_without_urls, width=100,
                                               break_long_words=False)
                    
                    # Restore URLs
                    for wrapped_line in wrapped_text:
                        restored = wrapped_line
                        for i, url in enumerate(urls):
                            restored = restored.replace(f'__URL_{i}__', url)
                        formatted_lines.append(restored)
                    continue
        
        # Regular paragraphs - wrap for better readability
        # Pandoc will handle wrapping, but we can format long paragraphs
        if len(stripped) > 100 and not in_list:
            # Wrap long paragraphs (but preserve structure)
            wrapped = textwrap.wrap(stripped, width=100, 
                                  break_long_words=False,
                                  break_on_hyphens=False)
            formatted_lines.extend(wrapped)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


__all__ = ['research_to_pdf_tool']
