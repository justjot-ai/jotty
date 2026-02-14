"""
Notion Research Documentation Skill - Research and document topics in Notion.

Searches across Notion workspace, synthesizes findings, and creates
comprehensive research documentation with citations.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("notion-research-documentation")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def research_and_document_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research topic in Notion and create documentation.
    
    Args:
        params:
            - research_topic (str): Topic to research
            - output_format (str, optional): Format type
            - search_queries (list, optional): Additional queries
            - parent_page_id (str, optional): Parent page
            - include_citations (bool, optional): Include citations
    
    Returns:
        Dictionary with documentation page and sources
    """
    status.set_callback(params.pop('_status_callback', None))

    research_topic = params.get('research_topic', '')
    output_format = params.get('output_format', 'summary')
    search_queries = params.get('search_queries', [])
    parent_page_id = params.get('parent_page_id', None)
    include_citations = params.get('include_citations', True)
    
    if not research_topic:
        return {
            'success': False,
            'error': 'research_topic is required'
        }
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        notion_skill = registry.get_skill('notion')
        
        if not notion_skill:
            return {
                'success': False,
                'error': 'notion skill not available'
            }
        
        # Search Notion
        search_tool = notion_skill.tools.get('search_tool') or notion_skill.tools.get('search_pages_tool')
        
        sources = []
        all_results = []
        
        queries = [research_topic] + search_queries
        
        for query in queries:
            if search_tool:
                if inspect.iscoroutinefunction(search_tool):
                    result = await search_tool({'query': query})
                else:
                    result = search_tool({'query': query})
                
                if result.get('success'):
                    results = result.get('results', [])
                    all_results.extend(results)
                    sources.extend([r.get('id', '') for r in results if r.get('id')])
        
        # Fetch detailed content for top sources
        fetch_tool = notion_skill.tools.get('fetch_tool') or notion_skill.tools.get('fetch_page_tool')
        
        source_content = []
        for source_id in sources[:5]:  # Top 5 sources
            if fetch_tool:
                if inspect.iscoroutinefunction(fetch_tool):
                    result = await fetch_tool({'page_id': source_id})
                else:
                    result = fetch_tool({'page_id': source_id})
                
                if result.get('success'):
                    source_content.append({
                        'id': source_id,
                        'title': result.get('title', ''),
                        'content': result.get('content', '')
                    })
        
        # Synthesize findings
        documentation_content = _synthesize_research(
            research_topic, source_content, output_format, include_citations
        )
        
        # Create documentation page
        create_tool = notion_skill.tools.get('create_page_tool') or notion_skill.tools.get('create_pages_tool')
        
        if not create_tool:
            return {
                'success': False,
                'error': 'Notion create page tool not found'
            }
        
        create_params = {
            'title': f"Research: {research_topic}",
            'content': documentation_content
        }
        
        if parent_page_id:
            create_params['parent_page_id'] = parent_page_id
        
        if inspect.iscoroutinefunction(create_tool):
            result = await create_tool(create_params)
        else:
            result = create_tool(create_params)
        
        if result.get('success'):
            return {
                'success': True,
                'documentation_page_id': result.get('page_id', ''),
                'sources_found': len(sources),
                'sources': sources[:10]  # Top 10
            }
        
        return {
            'success': False,
            'error': result.get('error', 'Failed to create documentation')
        }
        
    except Exception as e:
        logger.error(f"Research documentation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def _synthesize_research(
    topic: str,
    sources: List[Dict],
    output_format: str,
    include_citations: bool
) -> str:
    """Synthesize research findings into documentation."""
    
    if output_format == 'brief':
        content = f"""# {topic} - Quick Brief

## Summary

{_extract_key_points(sources)}

## Key Findings

{_format_findings(sources[:3])}

"""
    
    elif output_format == 'comprehensive':
        content = f"""# {topic} - Comprehensive Research Report

## Executive Summary

{_extract_key_points(sources)}

## Detailed Findings

{_format_findings(sources)}

## Analysis

[Analysis of findings and patterns]

## Conclusions

[Key conclusions and insights]

## Recommendations

[Actionable recommendations]

"""
    
    else:  # summary
        content = f"""# {topic} - Research Summary

## Overview

{_extract_key_points(sources)}

## Key Findings

{_format_findings(sources[:5])}

## Sources

{_format_citations(sources) if include_citations else ''}

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    
    return content


def _extract_key_points(sources: List[Dict]) -> str:
    """Extract key points from sources."""
    
    if not sources:
        return "No sources found."
    
    points = []
    for source in sources[:3]:
        content = source.get('content', '')
        # Extract first few sentences
        sentences = content.split('.')[:2]
        if sentences:
            points.append(' '.join(sentences).strip())
    
    return '\n\n'.join(f"- {point}" for point in points if point)


def _format_findings(sources: List[Dict]) -> str:
    """Format findings from sources."""
    
    formatted = ""
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Untitled')
        content = source.get('content', '')[:200]  # First 200 chars
        
        formatted += f"### {i}. {title}\n\n"
        formatted += f"{content}...\n\n"
    
    return formatted


def _format_citations(sources: List[Dict]) -> str:
    """Format citations."""
    
    citations = []
    for source in sources:
        title = source.get('title', 'Untitled')
        source_id = source.get('id', '')
        citations.append(f"- [[{source_id}|{title}]]")
    
    return '\n'.join(citations) if citations else ""
