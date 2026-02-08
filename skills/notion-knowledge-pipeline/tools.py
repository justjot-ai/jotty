"""
Notion Knowledge Pipeline Composite Skill

Complete knowledge management workflow:
1. Capture knowledge (notion-knowledge-capture)
2. Research and document (notion-research-documentation)
3. Create implementation plan (notion-spec-to-implementation)

Source → Processor → Sink pattern.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def notion_knowledge_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete Notion knowledge management workflow: capture → research → implement.
    
    Args:
        params:
            - workflow_type (str, required): 'capture', 'research', 'implement', or 'full'
            - knowledge_content (str, optional): Content to capture
            - knowledge_title (str, optional): Title for knowledge entry
            - content_type (str, optional): 'concept', 'meeting', 'idea', 'note' (default: 'concept')
            - research_topic (str, optional): Topic for research
            - research_output_format (str, optional): 'brief', 'detailed', 'comprehensive' (default: 'detailed')
            - spec_page_id (str, optional): Notion page ID for spec
            - plan_type (str, optional): 'quick', 'detailed', 'comprehensive' (default: 'quick')
            - capture_knowledge (bool, optional): Capture knowledge (default: True for 'full')
            - research_and_document (bool, optional): Research and document (default: True for 'full')
            - create_implementation_plan (bool, optional): Create plan (default: True for 'full')
    
    Returns:
        Dictionary with knowledge page, research page, and implementation plan
    """
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        workflow_type = params.get('workflow_type', 'full')
        
        logger.info(f"📚 Notion Knowledge Pipeline: {workflow_type}")
        
        knowledge_page_id = None
        research_page_id = None
        implementation_plan_id = None
        
        # Step 1: Capture knowledge (Source)
        if workflow_type in ['capture', 'full'] and params.get('capture_knowledge', workflow_type == 'full'):
            logger.info("💾 Step 1: Capturing knowledge...")
            capture_skill = registry.get_skill('notion-knowledge-capture')
            if capture_skill:
                capture_tool = capture_skill.tools.get('capture_knowledge_to_notion_tool')
                if capture_tool:
                    knowledge_content = params.get('knowledge_content')
                    if not knowledge_content:
                        logger.warning("⚠️  No knowledge_content provided, skipping capture")
                    else:
                        capture_params = {
                            'content': knowledge_content,
                            'title': params.get('knowledge_title', 'Knowledge Entry'),
                            'content_type': params.get('content_type', 'concept')
                        }
                        
                        if inspect.iscoroutinefunction(capture_tool):
                            capture_result = await capture_tool(capture_params)
                        else:
                            capture_result = capture_tool(capture_params)
                        
                        if capture_result.get('success'):
                            knowledge_page_id = capture_result.get('page_id')
                            logger.info(f"✅ Knowledge captured: {knowledge_page_id}")
                        else:
                            logger.warning(f"⚠️  Knowledge capture failed: {capture_result.get('error')}")
        
        # Step 2: Research and document (Processor)
        if workflow_type in ['research', 'full'] and params.get('research_and_document', workflow_type == 'full'):
            logger.info("🔍 Step 2: Researching and documenting...")
            research_skill = registry.get_skill('notion-research-documentation')
            if research_skill:
                research_tool = research_skill.tools.get('research_and_document_tool')
                if research_tool:
                    research_topic = params.get('research_topic')
                    if not research_topic:
                        logger.warning("⚠️  No research_topic provided, skipping research")
                    else:
                        research_params = {
                            'research_topic': research_topic,
                            'output_format': params.get('research_output_format', 'detailed')
                        }
                        
                        if inspect.iscoroutinefunction(research_tool):
                            research_result = await research_tool(research_params)
                        else:
                            research_result = research_tool(research_params)
                        
                        if research_result.get('success'):
                            research_page_id = research_result.get('page_id')
                            logger.info(f"✅ Research documented: {research_page_id}")
                        else:
                            logger.warning(f"⚠️  Research documentation failed: {research_result.get('error')}")
        
        # Step 3: Create implementation plan (Sink)
        if workflow_type in ['implement', 'full'] and params.get('create_implementation_plan', workflow_type == 'full'):
            logger.info("📋 Step 3: Creating implementation plan...")
            implement_skill = registry.get_skill('notion-spec-to-implementation')
            if implement_skill:
                implement_tool = implement_skill.tools.get('create_implementation_plan_tool')
                if implement_tool:
                    spec_page_id = params.get('spec_page_id')
                    if not spec_page_id:
                        logger.warning("⚠️  No spec_page_id provided, skipping implementation plan")
                    else:
                        implement_params = {
                            'spec_page_id': spec_page_id,
                            'plan_type': params.get('plan_type', 'quick')
                        }
                        
                        if inspect.iscoroutinefunction(implement_tool):
                            implement_result = await implement_tool(implement_params)
                        else:
                            implement_result = implement_tool(implement_params)
                        
                        if implement_result.get('success'):
                            implementation_plan_id = implement_result.get('plan_page_id')
                            logger.info(f"✅ Implementation plan created: {implementation_plan_id}")
                        else:
                            logger.warning(f"⚠️  Implementation plan creation failed: {implement_result.get('error')}")
        
        return {
            'success': True,
            'workflow_type': workflow_type,
            'knowledge_page_id': knowledge_page_id,
            'research_page_id': research_page_id,
            'implementation_plan_id': implementation_plan_id
        }
        
    except Exception as e:
        logger.error(f"Notion Knowledge Pipeline error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['notion_knowledge_pipeline_tool']
