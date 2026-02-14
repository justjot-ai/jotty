"""
Reddit Trending → Markdown → JustJot Idea Pipeline

Uses generic pipeline pattern:
- Source: web-search (Reddit trending topics)
- Processor: Format as markdown
- Sink: mcp-justjot (create idea)

DRY: Reuses existing skills, no duplication.
"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("reddit-trending-to-justjot")


logger = logging.getLogger(__name__)


def _format_reddit_results_as_markdown(results: list, topic: str) -> str:
    """Format Reddit search results as markdown."""
    lines = [
        f"# Reddit Trending: {topic}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** Reddit",
        f"**Results:** {len(results)} posts",
        f"",
        "---",
        f"",
    ]
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        snippet = result.get('snippet', '')
        
        lines.append(f"## {i}. {title}")
        lines.append("")
        if url:
            lines.append(f"**Link:** [{url}]({url})")
            lines.append("")
        if snippet:
            # Clean up snippet (remove extra whitespace)
            snippet_clean = ' '.join(snippet.split())
            lines.append(f"{snippet_clean}")
            lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


@async_tool_wrapper()
async def reddit_trending_to_justjot_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search Reddit for trending topics, format as markdown, and create JustJot idea.
    
    Args:
        params: Dictionary containing:
            - topic (str, required): Topic to search on Reddit
            - title (str, optional): Idea title (default: auto-generated)
            - max_results (int, optional): Max Reddit results (default: 10)
            - description (str, optional): Idea description
            - tags (list, optional): Tags for the idea
    
    Returns:
        Dictionary with:
            - success (bool): Whether workflow succeeded
            - idea_id (str): Created idea ID
            - idea (dict): Created idea data
            - markdown (str): Generated markdown content
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
        
        logger.info(f"🔍 Reddit Trending → Markdown → JustJot Idea: {topic}")
        
        # Step 1: Source - Search Reddit
        logger.info("📡 Step 1: Searching Reddit...")
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
        
        # Search Reddit specifically
        max_results = params.get('max_results', 10)
        search_query = f"{topic} site:reddit.com"
        
        import inspect
        if inspect.iscoroutinefunction(search_tool):
            search_result = await search_tool({
                'query': search_query,
                'max_results': max_results
            })
        else:
            search_result = search_tool({
                'query': search_query,
                'max_results': max_results
            })
        
        if not search_result.get('success'):
            return {
                'success': False,
                'error': f"Reddit search failed: {search_result.get('error')}"
            }
        
        results = search_result.get('results', [])
        if not results:
            return {
                'success': False,
                'error': 'No Reddit results found'
            }
        
        # Filter to Reddit URLs only
        reddit_results = [
            r for r in results 
            if 'reddit.com' in r.get('url', '').lower() or '/r/' in r.get('url', '')
        ]
        
        if not reddit_results:
            # Try alternative search
            search_query_alt = f"{topic} reddit"
            if inspect.iscoroutinefunction(search_tool):
                search_result_alt = await search_tool({
                    'query': search_query_alt,
                    'max_results': max_results
                })
            else:
                search_result_alt = search_tool({
                    'query': search_query_alt,
                    'max_results': max_results
                })
            
            if search_result_alt.get('success'):
                reddit_results = [
                    r for r in search_result_alt.get('results', [])
                    if 'reddit.com' in r.get('url', '').lower() or 'reddit' in r.get('title', '').lower()
                ]
        
        if not reddit_results:
            return {
                'success': False,
                'error': 'No Reddit results found after filtering'
            }
        
        logger.info(f"✅ Found {len(reddit_results)} Reddit posts")
        
        # Step 2: Processor - Format as markdown
        logger.info("📝 Step 2: Formatting as markdown...")
        markdown_content = _format_reddit_results_as_markdown(reddit_results, topic)
        
        logger.info(f"✅ Markdown generated: {len(markdown_content)} chars")
        
        # Step 3: Sink - Create JustJot idea
        logger.info("💡 Step 3: Creating JustJot idea...")
        
        # Try MCP client first (local), fallback to HTTP API (cmd.dev)
        use_mcp_client = params.get('use_mcp_client', True)
        idea_result = None
        
        if use_mcp_client:
            try:
                from Jotty.core.integration.mcp_client import MCPClient
                from pathlib import Path
                
                server_path = "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"
                if Path(server_path).exists():
                    logger.info("Using MCP client (stdio transport)...")
                    
                    # Generate idea title
                    title = params.get('title', f'Reddit Trends: {topic}')
                    description = params.get('description', f'Reddit trending topics about {topic}')
                    
                    async with MCPClient(server_path=server_path) as client:
                        mcp_result = await client.call_tool("create_idea", {
                            "title": title,
                            "description": description,
                            "tags": params.get('tags', ['reddit', 'trending', topic.lower()]),
                            "sections": [
                                {
                                    "title": "Reddit Trending Posts",
                                    "content": markdown_content,
                                    "type": "text"
                                }
                            ],
                            "status": "Draft"
                        })
                        
                        if mcp_result.get('isError'):
                            raise Exception(mcp_result.get('content', 'MCP error'))
                        
                        idea_data = mcp_result.get('content', {})
                        idea_result = {
                            'success': True,
                            'idea': idea_data,
                            'id': idea_data.get('_id') or idea_data.get('id')
                        }
                else:
                    logger.info("MCP server not found, falling back to HTTP API...")
                    use_mcp_client = False
            except Exception as e:
                logger.warning(f"MCP client failed: {e}, falling back to HTTP API...")
                use_mcp_client = False
        
        # Fallback to HTTP API
        if not use_mcp_client or not idea_result:
            logger.info("Using HTTP API...")
            mcp_skill = registry.get_skill('mcp-justjot')
            if not mcp_skill:
                return {
                    'success': False,
                    'error': 'mcp-justjot skill not available'
                }
            
            create_idea_tool = mcp_skill.tools.get('create_idea_tool')
            if not create_idea_tool:
                return {
                    'success': False,
                    'error': 'create_idea_tool not found'
                }
            
            # Generate idea title
            title = params.get('title', f'Reddit Trends: {topic}')
            description = params.get('description', f'Reddit trending topics about {topic}')
            
            # Create idea with markdown content as first section
            if inspect.iscoroutinefunction(create_idea_tool):
                idea_result = await create_idea_tool({
                    'title': title,
                    'description': description,
                    'tags': params.get('tags', ['reddit', 'trending', topic.lower()]),
                    'sections': [
                        {
                            'title': 'Reddit Trending Posts',
                            'content': markdown_content,
                            'type': 'text'  # Markdown text section
                        }
                    ],
                    'templateName': 'default',
                    'status': 'Draft'
                })
            else:
                idea_result = create_idea_tool({
                    'title': title,
                    'description': description,
                    'tags': params.get('tags', ['reddit', 'trending', topic.lower()]),
                    'sections': [
                        {
                            'title': 'Reddit Trending Posts',
                            'content': markdown_content,
                            'type': 'text'
                        }
                    ],
                    'templateName': 'default',
                    'status': 'Draft'
                })
        
        if not idea_result.get('success'):
            return {
                'success': False,
                'error': f"Idea creation failed: {idea_result.get('error')}"
            }
        
        idea = idea_result.get('idea', {})
        idea_id = idea_result.get('id') or idea.get('_id') or idea.get('id')
        
        logger.info(f"✅ Idea created: {idea_id}")
        
        return {
            'success': True,
            'idea_id': idea_id,
            'idea': idea,
            'markdown': markdown_content,
            'reddit_posts_count': len(reddit_results),
            'topic': topic,
            'title': title
        }
        
    except Exception as e:
        logger.error(f"Reddit Trending → JustJot Idea workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['reddit_trending_to_justjot_tool']
