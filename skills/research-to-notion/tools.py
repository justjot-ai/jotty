"""
Research → Content → Notion Composite Skill

Complete research workflow:
1. Research leads/competitors/topics (lead-research-assistant, competitive-ads-extractor)
2. Write content with research (content-research-writer)
3. Document in Notion (notion-research-documentation)

Source → Processor → Sink pattern.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("research-to-notion")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def research_to_notion_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete research workflow: research → write content → document in Notion.
    
    Args:
        params:
            - research_type (str, required): 'leads', 'competitive', or 'topic'
            - research_query (str, required): Query for research
            - product_description (str, optional): For lead research
            - competitor_name (str, optional): For competitive research
            - topic (str, optional): For topic research
            - content_action (str, optional): 'outline', 'draft', 'full' (default: 'outline')
            - notion_output_format (str, optional): 'brief', 'detailed', 'comprehensive' (default: 'detailed')
            - max_leads (int, optional): Max leads for lead research (default: 10)
            - max_ads (int, optional): Max ads for competitive research (default: 5)
            - create_notion_page (bool, optional): Create Notion page (default: True)
    
    Returns:
        Dictionary with research results, content, and Notion page info
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        research_type = params.get('research_type')
        research_query = params.get('research_query')
        
        if not research_type or not research_query:
            return {
                'success': False,
                'error': 'research_type and research_query are required'
            }
        
        logger.info(f"🔍 Research → Content → Notion workflow: {research_type} - {research_query}")
        
        # Step 1: Research (Source)
        logger.info(f"📡 Step 1: Researching ({research_type})...")
        research_results = None
        research_summary = None
        
        if research_type == 'leads':
            lead_skill = registry.get_skill('lead-research-assistant')
            if not lead_skill:
                return {'success': False, 'error': 'lead-research-assistant skill not available'}
            
            lead_tool = lead_skill.tools.get('research_leads_tool')
            if not lead_tool:
                return {'success': False, 'error': 'research_leads_tool not found'}
            
            lead_params = {
                'product_description': params.get('product_description', research_query),
                'max_leads': params.get('max_leads', 10),
                **{k: v for k, v in params.items() if k in ['industry', 'location', 'company_size', 'pain_points', 'technologies']}
            }
            
            if inspect.iscoroutinefunction(lead_tool):
                research_result = await lead_tool(lead_params)
            else:
                research_result = lead_tool(lead_params)
            
            if not research_result.get('success'):
                return {'success': False, 'error': f"Lead research failed: {research_result.get('error')}"}
            
            research_results = research_result.get('leads', [])
            research_summary = research_result.get('summary', '')
        
        elif research_type == 'competitive':
            ads_skill = registry.get_skill('competitive-ads-extractor')
            if not ads_skill:
                return {'success': False, 'error': 'competitive-ads-extractor skill not available'}
            
            ads_tool = ads_skill.tools.get('extract_competitive_ads_tool')
            if not ads_tool:
                return {'success': False, 'error': 'extract_competitive_ads_tool not found'}
            
            ads_params = {
                'competitor_name': params.get('competitor_name', research_query),
                'max_ads': params.get('max_ads', 5)
            }
            
            if inspect.iscoroutinefunction(ads_tool):
                research_result = await ads_tool(ads_params)
            else:
                research_result = ads_tool(ads_params)
            
            if not research_result.get('success'):
                return {'success': False, 'error': f"Competitive research failed: {research_result.get('error')}"}
            
            research_results = research_result.get('ads', [])
            research_summary = research_result.get('summary', '')
        
        else:  # topic research
            # Use web-search for general topic research
            web_search_skill = registry.get_skill('web-search')
            if not web_search_skill:
                return {'success': False, 'error': 'web-search skill not available'}
            
            search_tool = web_search_skill.tools.get('search_web_tool')
            if not search_tool:
                return {'success': False, 'error': 'search_web_tool not found'}
            
            if inspect.iscoroutinefunction(search_tool):
                research_result = await search_tool({
                    'query': research_query,
                    'max_results': 10
                })
            else:
                research_result = search_tool({
                    'query': research_query,
                    'max_results': 10
                })
            
            if not research_result.get('success'):
                return {'success': False, 'error': f"Topic research failed: {research_result.get('error')}"}
            
            research_results = research_result.get('results', [])
            research_summary = f"Found {len(research_results)} search results for: {research_query}"
        
        if not research_results:
            return {'success': False, 'error': 'No research results found'}
        
        logger.info(f"✅ Research complete: {len(research_results)} results")
        
        # Step 2: Write content with research (Processor)
        logger.info("✍️  Step 2: Writing content with research...")
        content_skill = registry.get_skill('content-research-writer')
        if not content_skill:
            return {'success': False, 'error': 'content-research-writer skill not available'}
        
        content_tool = content_skill.tools.get('write_content_with_research_tool')
        if not content_tool:
            return {'success': False, 'error': 'write_content_with_research_tool not found'}
        
        # Format research results for content writer
        research_context = f"Research Type: {research_type}\nQuery: {research_query}\n\n"
        if research_summary:
            research_context += f"Summary:\n{research_summary}\n\n"
        
        if isinstance(research_results, list):
            research_context += "Research Results:\n"
            for i, result in enumerate(research_results[:10], 1):
                if isinstance(result, dict):
                    title = result.get('title', result.get('name', result.get('company_name', '')))
                    desc = result.get('description', result.get('snippet', result.get('summary', '')))
                    research_context += f"{i}. {title}\n   {desc}\n\n"
                else:
                    research_context += f"{i}. {result}\n\n"
        
        content_action = params.get('content_action', 'outline')
        content_params = {
            'topic': research_query,
            'action': content_action,
            'draft_content': research_context,
            'research_topics': [research_query]
        }
        
        if inspect.iscoroutinefunction(content_tool):
            content_result = await content_tool(content_params)
        else:
            content_result = content_tool(content_params)
        
        if not content_result.get('success'):
            return {'success': False, 'error': f"Content writing failed: {content_result.get('error')}"}
        
        content_outline = content_result.get('outline', '')
        content_draft = content_result.get('draft', '')
        content_feedback = content_result.get('feedback', '')
        
        logger.info(f"✅ Content written: {content_action}")
        
        # Step 3: Document in Notion (Sink)
        notion_page_id = None
        if params.get('create_notion_page', True):
            logger.info("📝 Step 3: Documenting in Notion...")
            notion_skill = registry.get_skill('notion-research-documentation')
            if notion_skill:
                doc_tool = notion_skill.tools.get('research_and_document_tool')
                if doc_tool:
                    # Combine research and content for Notion
                    notion_content = f"# Research: {research_query}\n\n"
                    notion_content += f"**Research Type:** {research_type}\n\n"
                    if research_summary:
                        notion_content += f"## Summary\n{research_summary}\n\n"
                    if content_outline:
                        notion_content += f"## Content Outline\n{content_outline}\n\n"
                    if content_draft:
                        notion_content += f"## Draft Content\n{content_draft}\n\n"
                    if content_feedback:
                        notion_content += f"## Feedback\n{content_feedback}\n\n"
                    
                    output_format = params.get('notion_output_format', 'detailed')
                    doc_params = {
                        'research_topic': research_query,
                        'output_format': output_format,
                        'custom_content': notion_content
                    }
                    
                    if inspect.iscoroutinefunction(doc_tool):
                        notion_result = await doc_tool(doc_params)
                    else:
                        notion_result = doc_tool(doc_params)
                    
                    if notion_result.get('success'):
                        notion_page_id = notion_result.get('page_id')
                        logger.info(f"✅ Notion page created: {notion_page_id}")
                    else:
                        logger.warning(f"⚠️  Notion documentation failed: {notion_result.get('error')}")
        
        return {
            'success': True,
            'research_type': research_type,
            'research_query': research_query,
            'research_results_count': len(research_results) if isinstance(research_results, list) else 0,
            'research_summary': research_summary,
            'content_action': content_action,
            'content_outline': content_outline,
            'content_draft': content_draft,
            'content_feedback': content_feedback,
            'notion_page_id': notion_page_id,
            'notion_created': notion_page_id is not None
        }
        
    except Exception as e:
        logger.error(f"Research → Content → Notion workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['research_to_notion_tool']
