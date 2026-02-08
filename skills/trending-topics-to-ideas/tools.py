"""
Trending Topics → Ideas Composite Skill

For each trending topic:
1. Gather more details via HTTP/web search
2. Synthesize information via Claude CLI LLM
3. Create JustJot idea with synthesized content

Uses parallel processing for efficiency.
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("trending-topics-to-ideas")


logger = logging.getLogger(__name__)


def _format_topic_details_for_synthesis(results: list, topic: str) -> str:
    """Format web search results for synthesis."""
    lines = [
        f"# Information about: {topic}",
        f"",
        f"The following are web search results about '{topic}':",
        f"",
    ]
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        snippet = result.get('snippet', '')
        
        lines.append(f"## Source {i}: {title}")
        if url:
            lines.append(f"**URL:** {url}")
        if snippet:
            snippet_clean = ' '.join(snippet.split())
            lines.append(f"**Content:** {snippet_clean}")
        lines.append("")
    
    return "\n".join(lines)


async def _process_single_topic(
    topic: str,
    topic_index: int,
    registry,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single topic: gather details → synthesize → create idea.
    
    Returns:
        Dictionary with processing results
    """
    try:
        import inspect
        
        logger.info(f"📊 Processing topic {topic_index + 1}: {topic}")
        
        # Step 1: Gather details via web search
        logger.info(f"   🔍 Gathering details for: {topic}")
        web_search_skill = registry.get_skill('web-search')
        search_tool = web_search_skill.tools.get('search_web_tool')
        
        details_per_topic = params.get('details_per_topic', 5)
        if inspect.iscoroutinefunction(search_tool):
            search_result = await search_tool({
                'query': topic,
                'max_results': details_per_topic
            })
        else:
            search_result = search_tool({
                'query': topic,
                'max_results': details_per_topic
            })
        
        if not search_result.get('success'):
            return {
                'success': False,
                'topic': topic,
                'error': f"Search failed: {search_result.get('error')}"
            }
        
        details = search_result.get('results', [])
        if not details:
            return {
                'success': False,
                'topic': topic,
                'error': 'No search results found'
            }
        
        logger.info(f"   ✅ Found {len(details)} details")
        
        # Step 2: Synthesize via Claude CLI LLM
        logger.info(f"   🤖 Synthesizing information for: {topic}")
        claude_skill = registry.get_skill('claude-cli-llm')
        summarize_tool = claude_skill.tools.get('summarize_text_tool')
        
        # Format details for synthesis
        details_content = _format_topic_details_for_synthesis(details, topic)
        
        # Build synthesis prompt
        default_prompt = f"""Synthesize the following information about '{topic}'. 

Provide a comprehensive summary that:
1. Highlights key points and insights
2. Identifies trends and patterns
3. Notes important details and context
4. Organizes information in a clear, structured way

Format the synthesis with clear sections and bullet points where appropriate."""
        
        synthesize_prompt = params.get('synthesize_prompt', default_prompt)
        model = params.get('model', 'sonnet')
        
        if inspect.iscoroutinefunction(summarize_tool):
            synthesis_result = await summarize_tool({
                'content': details_content,
                'prompt': synthesize_prompt,
                'model': model
            })
        else:
            synthesis_result = summarize_tool({
                'content': details_content,
                'prompt': synthesize_prompt,
                'model': model
            })
        
        if not synthesis_result.get('success'):
            return {
                'success': False,
                'topic': topic,
                'error': f"Synthesis failed: {synthesis_result.get('error')}"
            }
        
        synthesized_content = synthesis_result.get('summary', '')
        if not synthesized_content:
            return {
                'success': False,
                'topic': topic,
                'error': 'No synthesis generated'
            }
        
        logger.info(f"   ✅ Synthesis complete: {len(synthesized_content)} chars")
        
        # Step 3: Create JustJot idea (if enabled)
        idea_id = None
        idea = None
        
        if params.get('create_ideas', True):
            logger.info(f"   💡 Creating idea for: {topic}")
            
            # Try MCP client first, fallback to HTTP API
            use_mcp_client = params.get('use_mcp_client', True)
            idea_result = None
            
            if use_mcp_client:
                try:
                    from Jotty.core.integration.mcp_client import MCPClient
                    from pathlib import Path
                    
                    server_path = "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"
                    if Path(server_path).exists():
                        async with MCPClient(server_path=server_path) as client:
                            mcp_result = await client.call_tool("create_idea", {
                                "title": f"Trending: {topic}",
                                "description": f"Synthesized information about trending topic: {topic}",
                                "tags": params.get('tags', ['trending', topic.lower().replace(' ', '-')]),
                                "sections": [
                                    {
                                        "title": "Synthesis",
                                        "content": synthesized_content,
                                        "type": "text"
                                    },
                                    {
                                        "title": "Sources",
                                        "content": _format_topic_details_for_synthesis(details, topic),
                                        "type": "text"
                                    }
                                ],
                                "status": "Draft"
                            })
                            
                            if not mcp_result.get('isError'):
                                idea_data = mcp_result.get('content', {})
                                idea_result = {
                                    'success': True,
                                    'idea': idea_data,
                                    'id': idea_data.get('_id') or idea_data.get('id')
                                }
                            else:
                                raise Exception(mcp_result.get('content', 'MCP error'))
                    else:
                        use_mcp_client = False
                except Exception as e:
                    logger.warning(f"MCP client failed for {topic}: {e}, falling back to HTTP API...")
                    use_mcp_client = False
            
            # Fallback to HTTP API
            if not use_mcp_client or not idea_result:
                mcp_skill = registry.get_skill('mcp-justjot')
                if mcp_skill:
                    create_idea_tool = mcp_skill.tools.get('create_idea_tool')
                    if create_idea_tool:
                        if inspect.iscoroutinefunction(create_idea_tool):
                            idea_result = await create_idea_tool({
                                'title': f'Trending: {topic}',
                                'description': f'Synthesized information about trending topic: {topic}',
                                'tags': params.get('tags', ['trending', topic.lower().replace(' ', '-')]),
                                'sections': [
                                    {
                                        'title': 'Synthesis',
                                        'content': synthesized_content,
                                        'type': 'text'
                                    },
                                    {
                                        'title': 'Sources',
                                        'content': _format_topic_details_for_synthesis(details, topic),
                                        'type': 'text'
                                    }
                                ],
                                'templateName': 'default',
                                'status': 'Draft'
                            })
                        else:
                            idea_result = create_idea_tool({
                                'title': f'Trending: {topic}',
                                'description': f'Synthesized information about trending topic: {topic}',
                                'tags': params.get('tags', ['trending', topic.lower().replace(' ', '-')]),
                                'sections': [
                                    {
                                        'title': 'Synthesis',
                                        'content': synthesized_content,
                                        'type': 'text'
                                    },
                                    {
                                        'title': 'Sources',
                                        'content': _format_topic_details_for_synthesis(details, topic),
                                        'type': 'text'
                                    }
                                ],
                                'templateName': 'default',
                                'status': 'Draft'
                            })
            
            if idea_result and idea_result.get('success'):
                idea = idea_result.get('idea', {})
                idea_id = idea_result.get('id') or idea.get('_id') or idea.get('id')
                logger.info(f"   ✅ Idea created: {idea_id}")
            else:
                logger.warning(f"   ⚠️  Idea creation failed for {topic}: {idea_result.get('error') if idea_result else 'Unknown error'}")
        
        return {
            'success': True,
            'topic': topic,
            'topic_index': topic_index,
            'details_count': len(details),
            'synthesis': synthesized_content,
            'idea_id': idea_id,
            'idea': idea
        }
        
    except Exception as e:
        logger.error(f"Error processing topic {topic}: {e}", exc_info=True)
        return {
            'success': False,
            'topic': topic,
            'error': f'Processing failed: {str(e)}'
        }


async def trending_topics_to_ideas_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get trending topics, process each: gather details → synthesize → create ideas.
    
    Args:
        params: Dictionary containing:
            - source (str, optional): 'reddit', 'v2v', or 'web' (default: 'reddit')
            - query (str, optional): Search query for trending topics
            - max_topics (int, optional): Max topics to process (default: 5)
            - details_per_topic (int, optional): Web search results per topic (default: 5)
            - create_ideas (bool, optional): Create JustJot ideas (default: True)
            - synthesize_prompt (str, optional): Custom synthesis prompt
            - use_mcp_client (bool, optional): Use MCP client (default: True)
            - model (str, optional): Claude model (default: 'sonnet')
            - tags (list, optional): Tags for ideas
    
    Returns:
        Dictionary with:
            - success (bool): Whether workflow succeeded
            - topics_processed (int): Number of topics processed
            - ideas_created (int): Number of ideas created
            - results (list): Results for each topic
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        source = params.get('source', 'reddit')
        query = params.get('query', 'trending topics')
        max_topics = params.get('max_topics', 5)
        
        logger.info(f"🔍 Trending Topics → Ideas workflow")
        logger.info(f"   Source: {source}")
        logger.info(f"   Query: {query}")
        logger.info(f"   Max topics: {max_topics}")
        
        # Step 1: Get trending topics
        logger.info("📡 Step 1: Getting trending topics...")
        topics = []
        
        if source == 'reddit':
            # Search Reddit for trending topics
            web_search_skill = registry.get_skill('web-search')
            search_tool = web_search_skill.tools.get('search_web_tool')
            
            import inspect
            search_query = f"{query} site:reddit.com" if query != 'trending topics' else "trending reddit"
            
            if inspect.iscoroutinefunction(search_tool):
                search_result = await search_tool({
                    'query': search_query,
                    'max_results': max_topics * 2  # Get more to filter
                })
            else:
                search_result = search_tool({
                    'query': search_query,
                    'max_results': max_topics * 2
                })
            
            if search_result.get('success'):
                results = search_result.get('results', [])
                # Extract topics from Reddit post titles
                for result in results:
                    title = result.get('title', '')
                    if title and 'reddit.com' in result.get('url', '').lower():
                        # Clean up title (remove "r/subreddit" prefixes, etc.)
                        topic = title.split(' - ')[0].split(' : ')[0].strip()
                        if topic and topic not in topics:
                            topics.append(topic)
                            if len(topics) >= max_topics:
                                break
        
        elif source == 'v2v':
            # Use V2V trending search
            v2v_skill = registry.get_skill('v2v-trending-search')
            if v2v_skill:
                search_tool = v2v_skill.tools.get('search_v2v_trending_tool')
                if search_tool:
                    import inspect
                    if inspect.iscoroutinefunction(search_tool):
                        v2v_result = await search_tool({
                            'query': query,
                            'max_results': max_topics
                        })
                    else:
                        v2v_result = search_tool({
                            'query': query,
                            'max_results': max_topics
                        })
                    
                    if v2v_result.get('success'):
                        results = v2v_result.get('results', [])
                        topics = [r.get('title', '') for r in results[:max_topics] if r.get('title')]
        
        else:  # web
            # General web search for trending topics
            web_search_skill = registry.get_skill('web-search')
            search_tool = web_search_skill.tools.get('search_web_tool')
            
            import inspect
            if inspect.iscoroutinefunction(search_tool):
                search_result = await search_tool({
                    'query': f"{query} trending 2026",
                    'max_results': max_topics
                })
            else:
                search_result = search_tool({
                    'query': f"{query} trending 2026",
                    'max_results': max_topics
                })
            
            if search_result.get('success'):
                results = search_result.get('results', [])
                topics = [r.get('title', '') for r in results[:max_topics] if r.get('title')]
        
        if not topics:
            return {
                'success': False,
                'error': f'No trending topics found from source: {source}'
            }
        
        topics = topics[:max_topics]  # Limit to max_topics
        logger.info(f"✅ Found {len(topics)} trending topics")
        for i, topic in enumerate(topics, 1):
            logger.info(f"   {i}. {topic}")
        
        # Step 2: Process each topic in parallel
        logger.info(f"🚀 Step 2: Processing {len(topics)} topics in parallel...")
        
        tasks = [
            _process_single_topic(topic, i, registry, params)
            for i, topic in enumerate(topics)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        ideas_created = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'topic': topics[i],
                    'error': str(result)
                })
            else:
                processed_results.append(result)
                if result.get('success') and result.get('idea_id'):
                    ideas_created += 1
        
        successful = sum(1 for r in processed_results if r.get('success'))
        
        logger.info(f"✅ Processing complete!")
        logger.info(f"   Topics processed: {successful}/{len(topics)}")
        logger.info(f"   Ideas created: {ideas_created}")
        
        return {
            'success': True,
            'topics_processed': len(topics),
            'topics_successful': successful,
            'ideas_created': ideas_created,
            'results': processed_results,
            'source': source
        }
        
    except Exception as e:
        logger.error(f"Trending Topics → Ideas workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['trending_topics_to_ideas_tool']
