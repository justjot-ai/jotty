"""
Transformer Paper Pipeline Skill

Generates a research paper on Transformers using Jotty's Unified LLM Provider,
converts it to PDF, and sends to Telegram.
"""
import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import sys

from Jotty.core.utils.skill_status import SkillStatus

# Add parent directory to path to import other skills

# Status emitter for progress updates
status = SkillStatus("transformer-paper-pipeline")

current_dir = Path(__file__).parent
jotty_root = current_dir.parent.parent
sys.path.insert(0, str(jotty_root))

try:
    from Jotty.core.registry.skills_registry import get_skills_registry
except ImportError:
    from Jotty.core.registry.skills_registry import get_skills_registry

logger = logging.getLogger(__name__)


async def generate_transformer_paper_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a transformer paper using Claude CLI LLM, convert to LaTeX PDF, and send to Telegram.
    
    Args:
        params: Dictionary containing:
            - topic (str, optional): Specific transformer topic (default: 'Transformers')
            - paper_length (str, optional): 'short', 'medium', or 'long' (default: 'medium')
            - include_math (bool, optional): Include mathematical equations (default: True)
            - output_dir (str, optional): Output directory (default: './output/transformer_papers')
            - send_telegram (bool, optional): Send PDF to Telegram (default: True)
            - telegram_chat_id (str, optional): Telegram chat ID (uses env var if not provided)
            - compile_pdf (bool, optional): Compile LaTeX to PDF (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - paper_content (str): Generated paper content
            - tex_file (str): Path to generated .tex file
            - pdf_file (str, optional): Path to generated PDF if compiled
            - telegram_sent (bool): Whether sent to Telegram
            - telegram_message_id (int, optional): Telegram message ID if sent
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        registry = get_skills_registry()
        if not registry.initialized:
            registry.init()
        
        topic = params.get('topic', 'Transformers')
        paper_length = params.get('paper_length', 'medium')
        include_math = params.get('include_math', True)
        output_dir = params.get('output_dir', './output/transformer_papers')
        send_telegram = params.get('send_telegram', True)
        compile_pdf = params.get('compile_pdf', True)
        
        # Step 1: Generate paper content using claude-cli-llm skill (uses unified LLM provider)
        logger.info(f"Step 1: Generating paper on {topic} using claude-cli-llm skill...")
        
        claude_skill = registry.get_skill('claude-cli-llm')
        if not claude_skill:
            return {
                'success': False,
                'error': 'claude-cli-llm skill not found'
            }
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return {
                'success': False,
                'error': 'generate_text_tool not found in claude-cli-llm skill'
            }
        
        # Create prompt for paper generation
        math_note = "Include LaTeX equations (use \\frac, \\sum, \\mathbf, etc.) for key formulas." if include_math else ""
        
        target_length_words = {
            'short': '2000-3000 words',
            'medium': '4000-6000 words',
            'long': '8000-12000 words'
        }
        
        max_tokens_map = {
            'short': 4000,
            'medium': 8000,
            'long': 16000
        }
        
        # Simplified, direct prompt - just ask for the paper content
        paper_prompt = f"""Write a complete research paper on {topic}. Write the actual paper content - not a description, not an outline, not a summary.

Write at least {target_length_words.get(paper_length, '4000-6000 words')} of actual paper content.

Start writing the paper now:

## Abstract

Write a 200-300 word abstract summarizing the key contributions and findings of {topic}.

## 1. Introduction

Write 5-7 full paragraphs covering:
- Historical context and background
- Motivation and problem statement  
- Key innovations and contributions
- Scope and organization

## 2. Architecture and Core Components

Write 8-10 full paragraphs explaining:
- Overall architecture design
- Encoder-decoder structure
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Layer normalization and residual connections
- Feed-forward networks

## 3. Mathematical Foundations

Write 6-8 full paragraphs with equations. {math_note}
Include formulas for:
- Scaled dot-product attention
- Multi-head attention
- Feed-forward networks
- Training objectives

## 4. Applications and Use Cases

Write 6-8 full paragraphs covering:
- NLP applications (BERT, GPT, T5)
- Computer Vision (ViT, DETR)
- Multimodal applications
- Real-world impact

## 5. Conclusion

Write 4-5 full paragraphs summarizing:
- Key contributions
- Current limitations
- Future research directions

Write the actual paper content now. Do not describe what you will write. Write full paragraphs with detailed explanations."""
        
        # Use generate_text_tool with API fallback for better max_tokens support
        import inspect
        is_async = inspect.iscoroutinefunction(generate_tool)
        
        tool_params = {
            'prompt': paper_prompt,
            'model': 'sonnet',
            'timeout': 600,  # 10 minutes
            'max_tokens': max_tokens_map.get(paper_length, 8000),
            'use_api': True  # Use API directly for better max_tokens control
        }
        
        if is_async:
            claude_result = await generate_tool(tool_params)
        else:
            claude_result = generate_tool(tool_params)
        
        if not claude_result.get('success'):
            return {
                'success': False,
                'error': f"Claude LLM generation failed: {claude_result.get('error')}"
            }
        
        # Extract content from result
        paper_content = (claude_result.get('text') or 
                        claude_result.get('summary') or 
                        claude_result.get('content') or 
                        claude_result.get('result', ''))
        
        if not paper_content:
            return {
                'success': False,
                'error': 'No content generated from Claude LLM'
            }
        
        logger.info(f"Generated paper content: {len(paper_content)} characters (~{len(paper_content)//5} words)")
        
        # Step 2: Create markdown document (same approach as stock research)
        logger.info("Step 2: Creating markdown document...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in topic)
        safe_topic = safe_topic.replace(' ', '_').lower()[:30]
        
        # Create comprehensive markdown report
        markdown_content = _create_markdown_paper(paper_content, topic, paper_length, include_math)
        
        # Save markdown
        markdown_file = Path(output_dir) / f'{safe_topic}_paper_{timestamp}.md'
        markdown_file.parent.mkdir(parents=True, exist_ok=True)
        markdown_file.write_text(markdown_content, encoding='utf-8')
        logger.info(f"Markdown saved: {markdown_file}")
        
        # Step 3: Convert markdown to PDF using document-converter (same as stock research)
        logger.info("Step 3: Converting markdown to PDF...")
        
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
        
        pdf_output_file = Path(output_dir) / f'{safe_topic}_paper_{timestamp}.pdf'
        
        pdf_result = convert_pdf_tool({
            'input_file': str(markdown_file),
            'output_file': str(pdf_output_file),
            'title': f'{topic}: A Comprehensive Research Paper',
            'author': 'Generated by Jotty Transformer Paper Pipeline',
            'page_size': 'a4'
        })
        
        if not pdf_result.get('success'):
            return {
                'success': False,
                'error': f'PDF conversion failed: {pdf_result.get("error")}',
                'paper_content': paper_content,
                'markdown_file': str(markdown_file)
            }
        
        # Get PDF path
        pdf_file = (
            pdf_result.get('output_path') or
            pdf_result.get('output_file') or
            pdf_result.get('pdf_path') or
            str(pdf_output_file) if pdf_output_file.exists() else None
        )
        
        if not pdf_file or not Path(pdf_file).exists():
            # Look for PDF in output directory
            pdf_files = list(Path(output_dir).glob(f'{safe_topic}_paper_*.pdf'))
            if pdf_files:
                pdf_file = str(pdf_files[-1])
            else:
                return {
                    'success': False,
                    'error': 'PDF file was not created',
                    'paper_content': paper_content,
                    'markdown_file': str(markdown_file)
                }
        
        logger.info(f"PDF generated: {pdf_file}")
        
        # Step 4: Send to Telegram
        telegram_sent = False
        telegram_message_id = None
        
        if send_telegram and pdf_file:
            logger.info("Step 4: Sending PDF to Telegram...")
            telegram_skill = registry.get_skill('telegram-sender')
            if telegram_skill:
                send_file_tool = telegram_skill.tools.get('send_telegram_file_tool')
                if send_file_tool:
                    telegram_chat_id = params.get('telegram_chat_id')
                    
                    import inspect
                    if inspect.iscoroutinefunction(send_file_tool):
                        telegram_result = await send_file_tool({
                            'file_path': pdf_file,
                            'chat_id': telegram_chat_id,
                            'caption': f"📄 Research Paper: {topic}\n\nGenerated via Claude CLI LLM and LaTeX Generator"
                        })
                    else:
                        telegram_result = send_file_tool({
                            'file_path': pdf_file,
                            'chat_id': telegram_chat_id,
                            'caption': f"📄 Research Paper: {topic}\n\nGenerated via Claude CLI LLM and LaTeX Generator"
                        })
                    
                    telegram_sent = telegram_result.get('success', False)
                    telegram_message_id = telegram_result.get('message_id')
                    
                    if telegram_sent:
                        logger.info(f"✅ PDF sent to Telegram (message_id: {telegram_message_id})")
                    else:
                        logger.warning(f"⚠️  Telegram send failed: {telegram_result.get('error')}")
        
        return {
            'success': True,
            'paper_content': paper_content,
            'markdown_file': str(markdown_file),
            'pdf_file': pdf_file,
            'telegram_sent': telegram_sent,
            'telegram_message_id': telegram_message_id
        }
        
    except Exception as e:
        logger.error(f"Error in transformer paper pipeline: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to generate transformer paper: {str(e)}'
        }


def _create_markdown_paper(content: str, topic: str, paper_length: str, include_math: bool) -> str:
    """Create markdown paper from Claude-generated content (similar to stock research approach)"""
    from datetime import datetime
    
    lines = []
    
    # Title
    lines.append(f"# {topic}: A Comprehensive Research Paper")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Paper Length:** {paper_length.title()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Parse content into sections
    sections = _parse_content_to_markdown_sections(content, topic, include_math)
    
    # Add sections to markdown
    for section in sections:
        title = section.get('title', '')
        content_blocks = section.get('content', [])
        
        if title:
            lines.append(f"## {title}")
            lines.append("")
        
        for block in content_blocks:
            block_type = block.get('type', 'text')
            
            if block_type == 'text':
                text_content = block.get('content', '')
                if text_content:
                    lines.append(text_content)
                    lines.append("")
            
            elif block_type == 'equation':
                latex = block.get('latex', '')
                if latex:
                    lines.append(f"$${latex}$$")
                    lines.append("")
            
            elif block_type == 'list':
                items = block.get('items', [])
                ordered = block.get('ordered', False)
                for i, item in enumerate(items, 1):
                    if ordered:
                        lines.append(f"{i}. {item}")
                    else:
                        lines.append(f"- {item}")
                lines.append("")
            
            elif block_type == 'keybox':
                title_box = block.get('title', 'Key Concept')
                content_box = block.get('content', '')
                lines.append(f"> **{title_box}**")
                lines.append(f"> {content_box}")
                lines.append("")
            
            elif block_type == 'example':
                title_ex = block.get('title', 'Example')
                content_ex = block.get('content', '')
                lines.append(f"**{title_ex}:**")
                lines.append(content_ex)
                lines.append("")
    
    return "\n".join(lines)


def _parse_content_to_markdown_sections(content: str, topic: str, include_math: bool) -> list:
    """
    Parse generated content into markdown sections.
    Improved parser that handles actual content (not just outlines).
    """
    """
    Parse generated content into LaTeX sections.
    Improved parser that handles various content formats.
    """
    sections = []
    
    # Standard section titles to look for
    section_keywords = {
        'abstract': 'Abstract',
        'introduction': 'Introduction',
        'background': 'Background',
        'architecture': 'Architecture',
        'architecture and core components': 'Architecture and Core Components',
        'mathematical foundations': 'Mathematical Foundations',
        'mathematics': 'Mathematical Foundations',
        'attention mechanism': 'Attention Mechanism',
        'applications': 'Applications',
        'applications and use cases': 'Applications and Use Cases',
        'use cases': 'Applications',
        'conclusion': 'Conclusion',
        'summary': 'Conclusion',
        'future work': 'Future Work'
    }
    
    # Split content by lines first to detect headers better
    lines = content.split('\n')
    current_section = None
    current_content_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_content_lines:
                current_content_lines.append('')
            continue
        
        line_lower = line_stripped.lower()
        
        # Check for markdown headers (# ## ###)
        is_header = False
        section_title = None
        
        if line_stripped.startswith('#'):
            # Markdown header
            header_level = len(line_stripped) - len(line_stripped.lstrip('#'))
            if header_level <= 2:  # Only process ## and ### as section headers
                section_title = line_stripped.lstrip('#').strip()
                is_header = True
        # Check for numbered sections (1. 2. etc.)
        elif line_stripped and line_stripped[0].isdigit() and '.' in line_stripped[:5]:
            # Could be a section header if it's short
            if len(line_stripped) < 100:
                potential_title = line_stripped.split('.', 1)[1].strip() if '.' in line_stripped else line_stripped
                # Check if it matches known section keywords
                for keyword, title in section_keywords.items():
                    if keyword in potential_title.lower():
                        section_title = title
                        is_header = True
                        break
        # Check for section keywords in the line
        else:
            for keyword, title in section_keywords.items():
                if keyword in line_lower and len(line_stripped) < 150:
                    # Make sure it's not part of a longer paragraph
                    if line_stripped == line_lower or line_stripped.istitle() or line_stripped.isupper():
                        section_title = title
                        is_header = True
                        break
        
        if is_header and section_title:
            # Save previous section
            if current_section and current_content_lines:
                content_text = '\n'.join(current_content_lines).strip()
                if content_text and len(content_text) > 50:  # Only add if substantial content
                    sections.append({
                        'title': current_section,
                        'level': 1,
                        'content': _parse_content_blocks_for_markdown(content_text, include_math)
                    })
            
            # Start new section
            current_section = section_title
            current_content_lines = []
        else:
            current_content_lines.append(line_stripped)
    
    # Add last section
    if current_section and current_content_lines:
        content_text = '\n'.join(current_content_lines).strip()
        if content_text and len(content_text) > 50:
            sections.append({
                'title': current_section,
                'level': 1,
                'content': _parse_content_blocks_for_markdown(content_text, include_math)
            })
    
    # If no sections found or content is too sparse, create better structure
    if not sections or sum(len(s['content']) for s in sections) < 500:
        # Try to parse as paragraphs and create sections
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        if len(paragraphs) >= 3:
            # Distribute paragraphs into logical sections
            sections = []
            if len(paragraphs) >= 1:
                sections.append({
                    'title': 'Abstract',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown('\n\n'.join(paragraphs[:1]), include_math)
                })
            if len(paragraphs) >= 2:
                sections.append({
                    'title': 'Introduction',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown('\n\n'.join(paragraphs[1:min(3, len(paragraphs))]), include_math)
                })
            if len(paragraphs) >= 4:
                mid_point = max(4, len(paragraphs) // 2)
                sections.append({
                    'title': 'Architecture and Core Components',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown('\n\n'.join(paragraphs[3:mid_point]), include_math)
                })
            if len(paragraphs) >= mid_point + 1:
                three_quarter = max(6, len(paragraphs) * 3 // 4)
                sections.append({
                    'title': 'Mathematical Foundations',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown('\n\n'.join(paragraphs[mid_point:three_quarter]), include_math)
                })
            if len(paragraphs) > three_quarter:
                sections.append({
                    'title': 'Applications and Conclusion',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown('\n\n'.join(paragraphs[three_quarter:]), include_math)
                })
        else:
            # Fallback: single comprehensive section
            sections = [
                {
                    'title': f'{topic} Overview',
                    'level': 1,
                    'content': _parse_content_blocks_for_markdown(content, include_math)
                }
            ]
    
    return sections


def _parse_content_blocks_for_markdown(text: str, include_math: bool) -> list:
    """Parse text into content blocks - improved to handle actual content"""
    blocks = []
    
    if not text or not text.strip():
        return blocks
    
    # Split into paragraphs (handle both \n\n and single \n)
    paragraphs = []
    current_para = []
    
    for line in text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        else:
            current_para.append(line_stripped)
    
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # If no paragraphs found, try splitting by double newlines
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        if not para or len(para) < 10:
            continue
        
        para_clean = para.strip()
        
        # Check for equations (LaTeX content)
        if include_math and '\\' in para_clean and any(cmd in para_clean for cmd in ['frac', 'sum', 'int', 'sqrt', 'mathbf', 'text', 'textcolor']):
            # Check if it's a standalone equation or inline
            if para_clean.count('\\') >= 2 and len(para_clean) < 500:
                # Likely a standalone equation
                blocks.append({
                    'type': 'equation',
                    'latex': para_clean,
                    'numbered': True,
                    'boxed': False
                })
            else:
                # Equation within text
                blocks.append({
                    'type': 'text',
                    'content': para_clean
                })
        # Check for lists (bullet points or numbered)
        elif (para_clean.startswith('-') or para_clean.startswith('*') or 
              (len(para_clean) > 2 and para_clean[0].isdigit() and para_clean[1:3] in ['. ', ') '])):
            # Extract list items
            items = []
            for item in para_clean.split('\n'):
                item_clean = item.strip().lstrip('-*').strip()
                # Remove numbering
                if item_clean and item_clean[0].isdigit():
                    parts = item_clean.split('.', 1)
                    if len(parts) > 1:
                        item_clean = parts[1].strip()
                    else:
                        parts = item_clean.split(')', 1)
                        if len(parts) > 1:
                            item_clean = parts[1].strip()
                
                if item_clean and len(item_clean) > 5:
                    items.append(item_clean)
            
            if len(items) > 1:
                blocks.append({
                    'type': 'list',
                    'ordered': para_clean[0].isdigit() if para_clean else False,
                    'items': items
                })
            else:
                # Single item, treat as text
                blocks.append({
                    'type': 'text',
                    'content': para_clean
                })
        # Check for key concepts
        elif any(keyword in para_clean.lower() for keyword in ['key concept', 'important note', 'note:', 'remember:', 'key point']):
            title = 'Key Concept'
            content = para_clean
            
            if ':' in para_clean:
                parts = para_clean.split(':', 1)
                if len(parts[0]) < 50:  # Reasonable title length
                    title = parts[0].strip()
                    content = parts[1].strip()
            
            blocks.append({
                'type': 'keybox',
                'title': title,
                'content': content
            })
        # Regular text paragraph
        else:
            # Skip if it looks like an outline item (too short, starts with number/bullet)
            # Only include substantial paragraphs
            if len(para_clean) > 80:  # Minimum paragraph length
                blocks.append({
                    'type': 'text',
                    'content': para_clean
                })
            elif len(para_clean) > 30 and not para_clean[0].isdigit() and not para_clean.startswith('-'):
                # Short but not an outline item
                blocks.append({
                    'type': 'text',
                    'content': para_clean
                })
    
    return blocks


__all__ = ['generate_transformer_paper_tool']
