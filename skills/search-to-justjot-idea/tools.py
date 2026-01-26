"""
Search to JustJot Idea Skill

Multi-agent workflow:
1. Search web for topic information
2. Summarize using Claude CLI LLM
3. Create idea on JustJot.ai via MCP client
"""
import asyncio
import inspect
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def search_and_create_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for information on a topic and create a JustJot idea.
    
    Multi-agent workflow:
    1. Source: Web search for topic
    2. Processor: Summarize with Claude CLI LLM
    3. Sink: Create idea on JustJot.ai via MCP client
    
    Args:
        params: Dictionary containing:
            - topic (str, required): Search topic/query
            - title (str, optional): Idea title (auto-generated if not provided)
            - description (str, optional): Idea description
            - tags (list, optional): Tags for the idea
            - max_results (int, optional): Maximum search results (default: 10)
            - summary_length (str, optional): 'brief', 'comprehensive', 'detailed' (default: 'comprehensive')
            - use_mcp_client (bool, optional): Use MCP client instead of HTTP API (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether idea was created
            - idea_id (str, optional): Created idea ID
            - title (str): Idea title
            - sections (int): Number of sections created
            - error (str, optional): Error message if failed
    """
    try:
        topic = params.get('topic')
        if not topic:
            return {
                'success': False,
                'error': 'topic parameter is required'
            }
        
        # Get parameters
        title = params.get('title')
        description = params.get('description', '')
        tags = params.get('tags', [])
        max_results = params.get('max_results', 10)
        summary_length = params.get('summary_length', 'comprehensive')
        use_mcp_client = params.get('use_mcp_client', True)
        
        logger.info(f"🔍 Starting search-to-idea workflow for topic: {topic}")
        
        # Step 1: Source - Web Search
        logger.info("📡 Step 1: Searching web for information...")
        from core.registry.skills_registry import get_skills_registry
        registry = get_skills_registry()
        registry.init()
        
        web_search_skill = registry.get_skill('web-search')
        if not web_search_skill:
            return {
                'success': False,
                'error': 'web-search skill not found'
            }
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        if not search_tool:
            return {
                'success': False,
                'error': 'search_web_tool not found'
            }
        
        # Perform search
        import inspect
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
                'error': f"Search failed: {search_result.get('error', 'Unknown error')}"
            }
        
        search_results = search_result.get('results', [])
        if not search_results:
            return {
                'success': False,
                'error': 'No search results found'
            }
        
        logger.info(f"   ✅ Found {len(search_results)} search results")
        
        # Format search results for summarization
        search_content = _format_search_results(search_results)
        
        # Step 2: Processor - Summarize with Claude CLI LLM
        logger.info("🤖 Step 2: Summarizing with Claude CLI LLM...")
        
        claude_skill = registry.get_skill('claude-cli-llm')
        if not claude_skill:
            return {
                'success': False,
                'error': 'claude-cli-llm skill not found'
            }
        
        summarize_tool = claude_skill.tools.get('summarize_text_tool')
        if not summarize_tool:
            return {
                'success': False,
                'error': 'summarize_text_tool not found'
            }
        
        # Create summary prompt
        summary_prompt = f"""Please provide a {summary_length} summary of the following information about "{topic}".

Organize the summary into clear sections with headings. Include:
1. Overview/Introduction
2. Key Concepts and Principles
3. Current Applications and Use Cases
4. Challenges and Limitations
5. Future Directions

Information:
{search_content}

Provide the summary in markdown format with clear sections."""

        if inspect.iscoroutinefunction(summarize_tool):
            summary_result = await summarize_tool({
                'content': summary_prompt,
                'target_length': summary_length
            })
        else:
            summary_result = summarize_tool({
                'content': summary_prompt,
                'target_length': summary_length
            })
        
        if not summary_result.get('success'):
            return {
                'success': False,
                'error': f"Summarization failed: {summary_result.get('error', 'Unknown error')}"
            }
        
        summary_text = summary_result.get('summary', summary_result.get('output', ''))
        logger.info(f"   ✅ Generated summary ({len(summary_text)} chars)")
        
        # Step 3: Sink - Create Idea on JustJot.ai
        logger.info("💡 Step 3: Creating idea on JustJot.ai...")
        
        # Try MCP client first, fallback to HTTP API
        idea_created = False
        idea_result = None
        
        if use_mcp_client:
            try:
                mcp_skill = registry.get_skill('mcp-justjot-mcp-client')
                if mcp_skill:
                    create_tool = mcp_skill.tools.get('create_idea_mcp_tool')
                    if create_tool:
                        logger.info("   Using MCP client (direct MongoDB)...")
                        idea_result = await create_tool({
                            'title': title or f"Research: {topic.title()}",
                            'description': description or f"Research and analysis on {topic}",
                            'tags': tags + ['research', 'ai-generated'],
                            'sections': _create_sections_from_summary(summary_text, search_results)
                        })
                        if idea_result.get('success'):
                            idea_created = True
            except Exception as e:
                logger.warning(f"   MCP client failed: {e}, falling back to HTTP API")
        
        # Fallback to HTTP API if MCP client failed
        if not idea_created:
            logger.info("   Using HTTP API...")
            http_skill = registry.get_skill('mcp-justjot')
            if http_skill:
                create_tool = http_skill.tools.get('create_idea_tool')
                if create_tool:
                    idea_result = await create_tool({
                        'title': title or f"Research: {topic.title()}",
                        'description': description or f"Research and analysis on {topic}",
                        'tags': tags + ['research', 'ai-generated'],
                        'sections': _create_sections_from_summary(summary_text, search_results)
                    })
                    if idea_result.get('success'):
                        idea_created = True
        
        if not idea_created:
            return {
                'success': False,
                'error': f"Idea creation failed: {idea_result.get('error', 'Unknown error') if idea_result else 'No create tool available'}"
            }
        
        idea_id = idea_result.get('id') or idea_result.get('idea', {}).get('id')
        sections = idea_result.get('sections', []) or _create_sections_from_summary(summary_text, search_results)
        
        logger.info(f"   ✅ Idea created successfully!")
        logger.info(f"      Idea ID: {idea_id}")
        logger.info(f"      Sections: {len(sections)}")
        
        return {
            'success': True,
            'idea_id': idea_id,
            'title': idea_result.get('title') or title or f"Research: {topic.title()}",
            'sections': len(sections),
            'tags': tags + ['research', 'ai-generated'],
            'method': 'mcp-client' if use_mcp_client and idea_created else 'http-api',
            'message': f'Idea "{title or topic.title()}" created successfully with {len(sections)} sections'
        }
        
    except Exception as e:
        logger.error(f"Search-to-idea workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for summarization."""
    formatted = []
    for i, result in enumerate(results[:10], 1):  # Limit to top 10
        title = result.get('title', 'No title')
        snippet = result.get('snippet', result.get('description', ''))
        url = result.get('url', '')
        formatted.append(f"\n[{i}] {title}\n{snippet}\nSource: {url}\n")
    return "\n".join(formatted)


def _create_sections_from_summary(summary_text: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create idea sections from summary text."""
    sections = []
    
    # Split summary into sections (by markdown headers)
    lines = summary_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        if line.startswith('# '):
            # Main title - skip or use as description
            continue
        elif line.startswith('## '):
            # Save previous section
            if current_section and current_content:
                sections.append({
                    'title': current_section,
                    'content': '\n'.join(current_content).strip(),
                    'type': 'text'
                })
            # Start new section
            current_section = line.replace('##', '').strip()
            current_content = []
        elif line.startswith('### '):
            # Subsection - add to current section
            if current_content:
                current_content.append('')
            current_content.append(line.replace('###', '**').strip() + '**')
        else:
            # Regular content
            if line.strip():
                current_content.append(line)
    
    # Add final section
    if current_section and current_content:
        sections.append({
            'title': current_section,
            'content': '\n'.join(current_content).strip(),
            'type': 'text'
        })
    
    # If no sections found, create a single summary section
    if not sections:
        sections.append({
            'title': 'Summary',
            'content': summary_text.strip(),
            'type': 'text'
        })
    
    # Add sources section
    if search_results:
        sources_content = []
        for i, result in enumerate(search_results[:10], 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            sources_content.append(f"{i}. [{title}]({url})")
        
        sections.append({
            'title': 'Sources',
            'content': '\n'.join(sources_content),
            'type': 'text'
        })
    
    return sections


__all__ = ['search_and_create_idea_tool']
