"""
Meeting Intelligence Pipeline Composite Skill

Complete meeting workflow:
1. Analyze meeting insights (meeting-insights-analyzer)
2. Prepare meeting materials (notion-meeting-intelligence)
3. Generate internal communications (internal-comms)

Source → Processor → Sink pattern.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any
from pathlib import Path

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("meeting-intelligence-pipeline")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def meeting_intelligence_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete meeting intelligence workflow: analyze → prepare → communicate.
    
    Args:
        params:
            - transcript_files (list, required): List of transcript file paths
            - user_name (str, required): User's name for analysis
            - meeting_topic (str, required): Meeting topic
            - meeting_type (str, optional): 'status_update', 'planning', 'retrospective', 'decision_making' (default: 'status_update')
            - analysis_types (list, optional): Analysis types (default: ['speaking_ratios', 'action_items', 'decisions'])
            - create_pre_read (bool, optional): Create pre-read (default: True)
            - create_agenda (bool, optional): Create agenda (default: True)
            - comm_type (str, optional): Communication type (default: '3p_update')
            - send_comm (bool, optional): Generate internal comm (default: True)
    
    Returns:
        Dictionary with analysis, meeting materials, and communication
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        transcript_files = params.get('transcript_files', [])
        user_name = params.get('user_name')
        meeting_topic = params.get('meeting_topic')
        
        if not transcript_files or not user_name or not meeting_topic:
            return {
                'success': False,
                'error': 'transcript_files, user_name, and meeting_topic are required'
            }
        
        logger.info(f"📊 Meeting Intelligence Pipeline: {meeting_topic}")
        
        # Step 1: Analyze meeting insights (Source)
        logger.info("📈 Step 1: Analyzing meeting insights...")
        insights_skill = registry.get_skill('meeting-insights-analyzer')
        if not insights_skill:
            return {'success': False, 'error': 'meeting-insights-analyzer skill not available'}
        
        insights_tool = insights_skill.tools.get('analyze_meeting_insights_tool')
        if not insights_tool:
            return {'success': False, 'error': 'analyze_meeting_insights_tool not found'}
        
        analysis_types = params.get('analysis_types', ['speaking_ratios', 'action_items', 'decisions'])
        insights_params = {
            'transcript_files': transcript_files,
            'user_name': user_name,
            'analysis_types': analysis_types
        }
        
        if inspect.iscoroutinefunction(insights_tool):
            insights_result = await insights_tool(insights_params)
        else:
            insights_result = insights_tool(insights_params)
        
        if not insights_result.get('success'):
            return {'success': False, 'error': f"Insights analysis failed: {insights_result.get('error')}"}
        
        insights = insights_result.get('insights', {})
        speaking_ratios = insights.get('speaking_ratios', {})
        action_items = insights.get('action_items', [])
        decisions = insights.get('decisions', [])
        
        logger.info(f"✅ Analysis complete: {len(action_items)} action items, {len(decisions)} decisions")
        
        # Step 2: Prepare meeting materials (Processor)
        logger.info("📋 Step 2: Preparing meeting materials...")
        meeting_skill = registry.get_skill('notion-meeting-intelligence')
        if not meeting_skill:
            return {'success': False, 'error': 'notion-meeting-intelligence skill not available'}
        
        meeting_tool = meeting_skill.tools.get('prepare_meeting_materials_tool')
        if not meeting_tool:
            return {'success': False, 'error': 'prepare_meeting_materials_tool not found'}
        
        meeting_type = params.get('meeting_type', 'status_update')
        meeting_params = {
            'meeting_topic': meeting_topic,
            'meeting_type': meeting_type,
            'create_pre_read': params.get('create_pre_read', True),
            'create_agenda': params.get('create_agenda', True),
            'action_items': action_items,
            'decisions': decisions
        }
        
        if inspect.iscoroutinefunction(meeting_tool):
            meeting_result = await meeting_tool(meeting_params)
        else:
            meeting_result = meeting_tool(meeting_params)
        
        if not meeting_result.get('success'):
            return {'success': False, 'error': f"Meeting preparation failed: {meeting_result.get('error')}"}
        
        pre_read_id = meeting_result.get('pre_read_page_id')
        agenda_id = meeting_result.get('agenda_page_id')
        
        logger.info(f"✅ Meeting materials prepared")
        
        # Step 3: Generate internal communication (Sink)
        comm_content = None
        if params.get('send_comm', True):
            logger.info("💬 Step 3: Generating internal communication...")
            comm_skill = registry.get_skill('internal-comms')
            if comm_skill:
                comm_tool = comm_skill.tools.get('write_internal_comm_tool')
                if comm_tool:
                    comm_type = params.get('comm_type', '3p_update')
                    
                    # Format action items and decisions for comm
                    progress = [f"Completed: {item}" for item in action_items[:3]] if action_items else ['Meeting completed']
                    plans = [f"Next: {item}" for item in action_items[3:6]] if len(action_items) > 3 else ['Follow up on decisions']
                    problems = decisions if decisions else ['No blockers']
                    
                    comm_params = {
                        'comm_type': comm_type,
                        'content': {
                            'progress': progress,
                            'plans': plans,
                            'problems': problems
                        }
                    }
                    
                    if inspect.iscoroutinefunction(comm_tool):
                        comm_result = await comm_tool(comm_params)
                    else:
                        comm_result = comm_tool(comm_params)
                    
                    if comm_result.get('success'):
                        comm_content = comm_result.get('content', '')
                        logger.info("✅ Internal communication generated")
                    else:
                        logger.warning(f"⚠️  Communication generation failed: {comm_result.get('error')}")
        
        return {
            'success': True,
            'meeting_topic': meeting_topic,
            'user_name': user_name,
            'speaking_ratios': speaking_ratios,
            'action_items': action_items,
            'decisions': decisions,
            'pre_read_page_id': pre_read_id,
            'agenda_page_id': agenda_id,
            'communication_content': comm_content,
            'materials_created': pre_read_id is not None or agenda_id is not None
        }
        
    except Exception as e:
        logger.error(f"Meeting Intelligence Pipeline error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['meeting_intelligence_pipeline_tool']
