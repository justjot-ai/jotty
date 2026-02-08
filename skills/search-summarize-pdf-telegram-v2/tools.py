"""
Search → Summarize → PDF → Telegram (Generic Pipeline Version)

Demonstrates generic Source → Processor → Sink pipeline pattern.
Uses declarative pipeline configuration.
"""
try:
    from Jotty.core.registry.pipeline_skill import create_pipeline_skill, StepType
except ImportError:
    from Jotty.core.registry.pipeline_skill import create_pipeline_skill, StepType

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("search-summarize-pdf-telegram-v2")

# Define pipeline declaratively

PIPELINE_CONFIG = [
    {
        "type": StepType.SOURCE.value,
        "skill": "web-search",
        "tool": "search_web_tool",
        "params": {
            "query": "{{topic}}",
            "max_results": "{{max_results}}"
        },
        "output_key": "source"
    },
    {
        "type": StepType.PROCESSOR.value,
        "skill": "claude-cli-llm",
        "tool": "summarize_text_tool",
        "params": lambda p, r: {
            "content": _format_search_results(r['source']['results'], p['topic']),
            "prompt": p.get('summarize_prompt', f"Summarize these search results about '{p['topic']}':"),
            "model": p.get('model', 'sonnet')
        },
        "output_key": "processor_summary"
    },
    {
        "type": StepType.PROCESSOR.value,
        "skill": "document-converter",
        "tool": "convert_to_pdf_tool",
        "params": lambda p, r: {
            "input_file": _create_temp_md(r['processor_summary']['summary'], p),
            "output_file": _get_output_path(p)
        },
        "output_key": "processor_pdf"
    },
    {
        "type": StepType.SINK.value,
        "skill": "telegram-sender",
        "tool": "send_telegram_file_tool",
        "params": lambda p, r: {
            "file_path": r['processor_pdf']['output_path'],
            "chat_id": p.get('telegram_chat_id'),
            "caption": f"📊 {p.get('title', 'Search Summary')}\n\nAI-generated summary using Claude CLI"
        },
        "output_key": "sink_telegram",
        "required": False  # Don't fail if Telegram send fails
    }
]


def _format_search_results(results: list, topic: str) -> str:
    """Format search results for summarization."""
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


def _create_temp_md(summary: str, params: dict) -> str:
    """Create temporary markdown file with summary."""
    import tempfile
    from pathlib import Path
    
    title = params.get('title', f"Summary: {params.get('topic', 'Unknown')}")
    markdown_content = f"# {title}\n\n"
    markdown_content += f"**Topic:** {params.get('topic')}\n"
    markdown_content += f"**Model:** {params.get('model', 'sonnet')}\n\n"
    markdown_content += "---\n\n"
    markdown_content += summary
    
    tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    tmp_file.write(markdown_content)
    tmp_file.close()
    
    return tmp_file.name


def _get_output_path(params: dict) -> str:
    """Get output PDF path."""
    from pathlib import Path
    from datetime import datetime
    
    output_dir = params.get('output_dir')
    if not output_dir:
        # Default to stock_market/outputs
        current_file = Path(__file__).resolve()
        jotty_dir = current_file.parent.parent.parent
        stock_market_root = jotty_dir.parent
        output_dir = stock_market_root / 'outputs'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title = params.get('title', f"Summary: {params.get('topic', 'Unknown')}")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_').lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f"{safe_title}_{timestamp}.pdf"
    
    return str(output_dir / pdf_filename)


# Create pipeline skill instance
_pipeline_skill = create_pipeline_skill(
    name='search-summarize-pdf-telegram-v2',
    description='Search → Summarize → PDF → Telegram (Generic Pipeline)',
    pipeline=PIPELINE_CONFIG
)


async def search_summarize_pdf_telegram_v2_tool(params: dict) -> dict:
    """
    Execute pipeline workflow.
    
    Uses generic pipeline framework with declarative configuration.
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_registry import get_skills_registry
    except ImportError:
        from Jotty.core.registry.skills_registry import get_skills_registry
    
    registry = get_skills_registry()
    registry.init()
    
    # Set defaults
    params.setdefault('max_results', 10)
    params.setdefault('model', 'sonnet')
    
    # Execute pipeline
    result = await _pipeline_skill.execute(params, registry)
    
    # Format response
    if result.get('_success'):
        return {
            'success': True,
            'pdf_path': result.get('processor_pdf', {}).get('output_path'),
            'telegram_sent': result.get('sink_telegram', {}).get('success', False),
            'summary': result.get('processor_summary', {}).get('summary'),
            'topic': params.get('topic'),
            'title': params.get('title')
        }
    else:
        return {
            'success': False,
            'error': result.get('_error', 'Pipeline execution failed')
        }


__all__ = ['search_summarize_pdf_telegram_v2_tool']
