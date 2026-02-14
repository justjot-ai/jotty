"""
V2V Trending Search Skill

Search for trending topics on V2V.ai and generate reports.
"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("v2v-trending-search")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def search_v2v_trending_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for trending topics on V2V.ai.
    
    Args:
        params: Dictionary containing:
            - query (str, optional): Search query (default: searches for trending)
            - format (str, optional): Output format - 'markdown', 'json' (default: 'markdown')
            - max_results (int, optional): Maximum results (default: 10)
    
    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - content (str): Search results in markdown format
            - results (list, optional): Raw results if format='json'
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        query = params.get('query', 'trending topics')
        output_format = params.get('format', 'markdown')
        max_results = params.get('max_results', 10)
        
        logger.info(f"Searching V2V.ai for: {query}")
        
        # Use web-search skill to search V2V.ai
        registry = get_skills_registry()
        registry.init()
        
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
        
        # Search V2V.ai - try different search strategies
        # V2V.ai might require authentication or have different URL structure
        search_queries = [
            f"site:v2v.ai {query}",
            f"v2v.ai {query}",
            f"{query} v2v",
            f"site:v2v.ai trending" if query == 'trending topics' else f"site:v2v.ai {query}"
        ]
        
        # Try first query, if no results try alternatives
        search_query = search_queries[0]
        
        # Check if search_tool is async
        import inspect
        
        # Try multiple search queries if first one fails
        results = []
        search_result = None
        
        for attempt_query in search_queries:
            if inspect.iscoroutinefunction(search_tool):
                search_result = await search_tool({
                    'query': attempt_query,
                    'max_results': max_results
                })
            else:
                search_result = search_tool({
                    'query': attempt_query,
                    'max_results': max_results
                })
            
            if search_result.get('success'):
                results = search_result.get('results', [])
                if results:
                    logger.info(f"✅ Found {len(results)} results with query: {attempt_query}")
                    break
            else:
                logger.debug(f"Query '{attempt_query}' failed: {search_result.get('error')}")
        
        if not results:
            # If no results, try general web search without site: restriction
            logger.info("Trying general web search for V2V...")
            if inspect.iscoroutinefunction(search_tool):
                search_result = await search_tool({
                    'query': f"v2v.ai {query}",
                    'max_results': max_results
                })
            else:
                search_result = search_tool({
                    'query': f"v2v.ai {query}",
                    'max_results': max_results
                })
            
            if search_result.get('success'):
                results = search_result.get('results', [])
                # Filter to V2V.ai URLs
                results = [r for r in results if 'v2v.ai' in r.get('url', '').lower()]
        
        if not results:
            return {
                'success': False,
                'error': f"No results found for V2V.ai search. V2V.ai may require authentication or the site structure may have changed.",
                'hint': 'Try using last30days-claude-cli skill instead for research'
            }
        
        if output_format == 'json':
            return {
                'success': True,
                'results': results,
                'count': len(results)
            }
        
        # Format as markdown
        markdown_content = f"# V2V.ai Trending Search Results\n\n"
        markdown_content += f"**Query:** {query}\n"
        markdown_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown_content += f"**Results:** {len(results)}\n\n"
        markdown_content += "---\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippet', result.get('description', 'No description'))
            
            markdown_content += f"## {i}. {title}\n\n"
            markdown_content += f"**URL:** {url}\n\n"
            markdown_content += f"{snippet}\n\n"
            markdown_content += "---\n\n"
        
        markdown_content += f"\n*Generated by Jotty V2V Trending Search Skill*\n"
        
        return {
            'success': True,
            'content': markdown_content,
            'results_count': len(results),
            'query': query
        }
        
    except Exception as e:
        logger.error(f"V2V search error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'V2V search failed: {str(e)}'
        }


__all__ = ['search_v2v_trending_tool']
