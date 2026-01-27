"""
Development Workflow Composite Skill

Complete development workflow:
1. Generate changelog (changelog-generator)
2. Create new skill (skill-creator)
3. Test webapp (webapp-testing)

Source → Processor → Sink pattern.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


async def dev_workflow_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete development workflow: changelog → skill creation → testing.
    
    Args:
        params:
            - workflow_type (str, required): 'changelog', 'skill_creation', 'testing', or 'full'
            - changelog_since (str, optional): Git reference for changelog (default: 'HEAD~10')
            - changelog_version (str, optional): Version for changelog
            - skill_name (str, optional): Name for new skill
            - skill_description (str, optional): Description for new skill
            - app_url (str, optional): URL for webapp testing (default: 'http://localhost:3000')
            - test_type (str, optional): Test type (default: 'screenshot')
            - generate_changelog (bool, optional): Generate changelog (default: True for 'full')
            - create_skill (bool, optional): Create skill (default: True for 'full')
            - test_app (bool, optional): Test app (default: True for 'full')
    
    Returns:
        Dictionary with changelog, skill info, and test results
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        workflow_type = params.get('workflow_type', 'full')
        
        logger.info(f"🔧 Development Workflow: {workflow_type}")
        
        changelog_result = None
        skill_result = None
        test_result = None
        
        # Step 1: Generate changelog (Source)
        if workflow_type in ['changelog', 'full'] and params.get('generate_changelog', workflow_type == 'full'):
            logger.info("📝 Step 1: Generating changelog...")
            changelog_skill = registry.get_skill('changelog-generator')
            if changelog_skill:
                changelog_tool = changelog_skill.tools.get('generate_changelog_tool')
                if changelog_tool:
                    changelog_params = {
                        'since': params.get('changelog_since', 'HEAD~10'),
                        'version': params.get('changelog_version', '1.0.0')
                    }
                    
                    if inspect.iscoroutinefunction(changelog_tool):
                        changelog_result = await changelog_tool(changelog_params)
                    else:
                        changelog_result = changelog_tool(changelog_params)
                    
                    if changelog_result.get('success'):
                        logger.info(f"✅ Changelog generated: {changelog_result.get('changelog_path')}")
                    else:
                        logger.warning(f"⚠️  Changelog generation failed: {changelog_result.get('error')}")
        
        # Step 2: Create skill (Processor)
        if workflow_type in ['skill_creation', 'full'] and params.get('create_skill', workflow_type == 'full'):
            logger.info("🛠️  Step 2: Creating skill...")
            skill_creator_skill = registry.get_skill('skill-creator')
            if skill_creator_skill:
                create_tool = skill_creator_skill.tools.get('create_skill_template_tool')
                if create_tool:
                    skill_name = params.get('skill_name')
                    skill_description = params.get('skill_description', 'A new Jotty skill')
                    
                    if not skill_name:
                        return {
                            'success': False,
                            'error': 'skill_name is required for skill creation'
                        }
                    
                    skill_params = {
                        'skill_name': skill_name,
                        'description': skill_description,
                        'include_tools': True
                    }
                    
                    if inspect.iscoroutinefunction(create_tool):
                        skill_result = await create_tool(skill_params)
                    else:
                        skill_result = create_tool(skill_params)
                    
                    if skill_result.get('success'):
                        logger.info(f"✅ Skill created: {skill_result.get('skill_path')}")
                    else:
                        logger.warning(f"⚠️  Skill creation failed: {skill_result.get('error')}")
        
        # Step 3: Test webapp (Sink)
        if workflow_type in ['testing', 'full'] and params.get('test_app', workflow_type == 'full'):
            logger.info("🧪 Step 3: Testing webapp...")
            testing_skill = registry.get_skill('webapp-testing')
            if testing_skill:
                test_tool = testing_skill.tools.get('test_webapp_tool')
                if test_tool:
                    app_url = params.get('app_url', 'http://localhost:3000')
                    test_params = {
                        'app_url': app_url,
                        'test_type': params.get('test_type', 'screenshot')
                    }
                    
                    if inspect.iscoroutinefunction(test_tool):
                        test_result = await test_tool(test_params)
                    else:
                        test_result = test_tool(test_params)
                    
                    if test_result.get('success'):
                        logger.info(f"✅ Webapp tested: {app_url}")
                    else:
                        logger.warning(f"⚠️  Webapp testing failed: {test_result.get('error')}")
        
        return {
            'success': True,
            'workflow_type': workflow_type,
            'changelog': changelog_result,
            'skill': skill_result,
            'test': test_result
        }
        
    except Exception as e:
        logger.error(f"Development Workflow error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['dev_workflow_tool']
