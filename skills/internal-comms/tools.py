"""
Internal Communications Skill - Write internal comms using standard formats.

Provides templates and guidelines for various internal communication types
including 3P updates, newsletters, FAQs, status reports, and more.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("internal-comms")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def write_internal_comm_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write internal communications using standard formats.
    
    Args:
        params:
            - comm_type (str): Type of communication
            - content (dict): Content data
            - format (str, optional): Output format
            - tone (str, optional): Writing tone
    
    Returns:
        Dictionary with generated communication
    """
    status.set_callback(params.pop('_status_callback', None))

    comm_type = params.get('comm_type', '')
    content = params.get('content', {})
    output_format = params.get('format', 'markdown')
    tone = params.get('tone', 'professional')
    
    if not comm_type:
        return {
            'success': False,
            'error': 'comm_type is required'
        }
    
    if not content:
        return {
            'success': False,
            'error': 'content is required'
        }
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if not claude_skill:
            return {
                'success': False,
                'error': 'claude-cli-llm skill not available'
            }
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return {
                'success': False,
                'error': 'generate_text_tool not found'
            }
        
        # Generate communication based on type
        prompt = _build_prompt(comm_type, content, tone)
        
        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 120
            })
        else:
            result = generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 120
            })
        
        if result.get('success'):
            communication = result.get('text', '')
            
            # Format output
            if output_format == 'html':
                communication = _markdown_to_html(communication)
            
            return {
                'success': True,
                'communication': communication,
                'format': output_format
            }
        
        return {
            'success': False,
            'error': 'Failed to generate communication'
        }
        
    except Exception as e:
        logger.error(f"Internal comm generation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def _build_prompt(comm_type: str, content: Dict, tone: str) -> str:
    """Build prompt based on communication type."""
    
    base_prompt = f"""Write a {tone} internal communication of type: {comm_type}

**Content Data:**
{_format_content(content)}

**Guidelines:**
"""
    
    if comm_type == '3p_update':
        return base_prompt + """
Format as:
- **Progress**: What we've accomplished
- **Plans**: What's coming next
- **Problems**: Challenges we're facing

Keep it concise, actionable, and transparent."""
    
    elif comm_type == 'newsletter':
        return base_prompt + """
Format as:
- Opening message
- Key highlights/announcements
- Team updates
- Upcoming events
- Closing note

Make it engaging and informative."""
    
    elif comm_type == 'faq':
        return base_prompt + """
Format as:
- Question
- Answer (clear and concise)
- Additional context if needed

Use Q&A format, be helpful and clear."""
    
    elif comm_type == 'status_report':
        return base_prompt + """
Format as:
- Project/Initiative name
- Current status (On track / At risk / Blocked)
- Key highlights
- Risks and blockers
- Next steps

Be factual and forward-looking."""
    
    elif comm_type == 'leadership_update':
        return base_prompt + """
Format as:
- Executive summary
- Key metrics/achievements
- Strategic initiatives
- Challenges and opportunities
- Looking ahead

Be strategic and inspiring."""
    
    elif comm_type == 'project_update':
        return base_prompt + """
Format as:
- Project status
- Completed milestones
- In progress work
- Upcoming milestones
- Blockers/risks

Be specific and actionable."""
    
    elif comm_type == 'incident_report':
        return base_prompt + """
Format as:
- Incident summary
- Timeline
- Impact assessment
- Root cause
- Resolution steps
- Prevention measures

Be thorough and transparent."""
    
    else:
        return base_prompt + "Write a clear, professional internal communication."


def _format_content(content: Dict) -> str:
    """Format content dict for prompt."""
    
    lines = []
    for key, value in content.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def _markdown_to_html(markdown: str) -> str:
    """Simple markdown to HTML conversion."""
    
    # Basic conversion (can be enhanced)
    html = markdown.replace('\n\n', '</p><p>')
    html = html.replace('\n', '<br>')
    html = f"<p>{html}</p>"
    
    # Headers
    for i in range(6, 0, -1):
        html = html.replace('#' * i + ' ', f'<h{i}>').replace('\n', f'</h{i}>', 1)
    
    # Bold
    html = html.replace('**', '<strong>').replace('**', '</strong>')
    
    # Lists
    html = html.replace('- ', '<li>')
    html = html.replace('<li>', '<ul><li>', 1)
    html += '</ul>'
    
    return html
