"""
Research to PDF Template

Common use case: Web search → Summarize → PDF report.

Workflow:
1. Parallel web searches (multiple queries)
2. Combine search results
3. Summarize content (optional)
4. Write to markdown
5. Convert to PDF

Customizable for:
- Number of searches
- Search queries
- Summarization options
- PDF formatting
"""
import logging
from typing import Dict, Any, List
import asyncio

logger = logging.getLogger(__name__)


async def research_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research topic and generate PDF report.
    
    Template for: Parallel Search → Combine → Summarize → PDF workflow
    
    Args:
        params: Dictionary containing:
            - topic (str, required): Research topic
            - queries (list, optional): Custom search queries (default: auto-generated)
            - num_searches (int, optional): Number of parallel searches (default: 3)
            - summarize (bool, optional): Summarize results (default: True)
            - output_file (str, optional): Output PDF path
            - max_results_per_search (int, optional): Results per search (default: 5)
    
    Returns:
        Dictionary with PDF path and research metadata
    """
    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry
        
        topic = params.get('topic', '')
        if not topic:
            return {
                'success': False,
                'error': 'topic parameter is required'
            }
        
        queries = params.get('queries', [])
        num_searches = params.get('num_searches', 3)
        summarize = params.get('summarize', True)
        output_file = params.get('output_file')
        max_results = params.get('max_results_per_search', 5)
        
        # Generate queries if not provided
        if not queries:
            queries = [
                f"{topic} overview",
                f"{topic} recent developments",
                f"{topic} key insights"
            ][:num_searches]
        
        # Auto-generate output file
        if not output_file:
            import tempfile
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            output_file = f"{safe_topic.replace(' ', '_')}_research.pdf"
        
        registry = get_skills_registry()
        registry.init()
        
        composer_skill = registry.get_skill('skill-composer')
        if not composer_skill:
            return {
                'success': False,
                'error': 'skill-composer not available'
            }
        
        compose_tool = composer_skill.tools.get('compose_skills_tool')
        
        # Build workflow
        workflow_steps = [
            {
                'type': 'parallel',
                'name': 'research',
                'skills': [
                    {
                        'skill': 'web-search',
                        'tool': 'search_web_tool',
                        'params': {'query': query, 'max_results': max_results}
                    }
                    for query in queries
                ]
            }
        ]
        
        # Add summarization if requested
        if summarize:
            workflow_steps.append({
                'type': 'single',
                'name': 'summarize',
                'skill': 'summarize',
                'tool': 'summarize_text_tool',
                'params': {
                    'text': '${research.output.results}',  # Will need to extract from parallel results
                    'max_length': 500
                }
            })
        
        # Add file writing
        import tempfile
        temp_md = tempfile.mktemp(suffix='.md', prefix='research_')
        
        workflow_steps.append({
            'type': 'single',
            'name': 'write_report',
            'skill': 'file-operations',
            'tool': 'write_file_tool',
            'params': {
                'path': temp_md,
                'content': f'# Research Report: {topic}\n\n## Search Results\n\n${research.output.results}'
            }
        })
        
        # Add PDF conversion
        workflow_steps.append({
            'type': 'single',
            'name': 'convert_pdf',
            'skill': 'document-converter',
            'tool': 'convert_to_pdf_tool',
            'params': {
                'input_file': '${write_report.path}',
                'output_file': output_file
            }
        })
        
        workflow = {'workflow': workflow_steps}
        
        # Execute workflow
        result = await compose_tool(workflow)
        
        if not result.get('success'):
            return {
                'success': False,
                'error': f'Research to PDF failed: {result.get("error")}'
            }
        
        import os
        return {
            'success': True,
            'pdf_path': output_file,
            'file_size': os.path.getsize(output_file) if os.path.exists(output_file) else 0,
            'topic': topic,
            'searches_performed': num_searches,
            'summarized': summarize
        }
        
    except Exception as e:
        logger.error(f"Research to PDF failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }
