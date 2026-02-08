"""
Notion Spec to Implementation Skill - Transform specs into implementation plans.

Reads specifications from Notion and creates structured implementation plans
with tasks, milestones, and progress tracking.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


async def create_implementation_plan_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create implementation plan from Notion specification.
    
    Args:
        params:
            - spec_page_id (str): Notion spec page ID
            - plan_type (str, optional): Type of plan
            - output_database_id (str, optional): Tasks database ID
            - include_milestones (bool, optional): Include milestones
            - breakdown_level (str, optional): Task breakdown level
    
    Returns:
        Dictionary with plan page and tasks
    """
    spec_page_id = params.get('spec_page_id', '')
    plan_type = params.get('plan_type', 'standard')
    output_database_id = params.get('output_database_id', None)
    include_milestones = params.get('include_milestones', True)
    breakdown_level = params.get('breakdown_level', 'medium')
    
    if not spec_page_id:
        return {
            'success': False,
            'error': 'spec_page_id is required'
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
        
        # Fetch specification page
        fetch_tool = notion_skill.tools.get('fetch_tool') or notion_skill.tools.get('fetch_page_tool')
        
        if not fetch_tool:
            return {
                'success': False,
                'error': 'Notion fetch tool not found'
            }
        
        if inspect.iscoroutinefunction(fetch_tool):
            spec_result = await fetch_tool({'page_id': spec_page_id})
        else:
            spec_result = fetch_tool({'page_id': spec_page_id})
        
        if not spec_result.get('success'):
            return {
                'success': False,
                'error': f"Failed to fetch spec page: {spec_result.get('error', 'Unknown error')}"
            }
        
        spec_content = spec_result.get('content', '')
        spec_title = spec_result.get('title', 'Specification')
        
        # Parse specification and create plan
        plan_content, tasks, milestones = await _create_plan_from_spec(
            spec_content, spec_title, plan_type, breakdown_level, include_milestones
        )
        
        # Create implementation plan page
        create_tool = notion_skill.tools.get('create_page_tool') or notion_skill.tools.get('create_pages_tool')
        
        if not create_tool:
            return {
                'success': False,
                'error': 'Notion create page tool not found'
            }
        
        plan_title = f"Implementation Plan: {spec_title}"
        
        create_params = {
            'title': plan_title,
            'content': plan_content
        }
        
        if inspect.iscoroutinefunction(create_tool):
            plan_result = await create_tool(create_params)
        else:
            plan_result = create_tool(create_params)
        
        if not plan_result.get('success'):
            return {
                'success': False,
                'error': f"Failed to create plan page: {plan_result.get('error', 'Unknown error')}"
            }
        
        plan_page_id = plan_result.get('page_id', '')
        
        # Create tasks in database if provided
        tasks_created = 0
        if output_database_id and tasks:
            add_tool = notion_skill.tools.get('add_to_database_tool') or notion_skill.tools.get('create_database_item_tool')
            
            if add_tool:
                for task in tasks:
                    task_params = {
                        'database_id': output_database_id,
                        'title': task.get('title', ''),
                        'properties': {
                            'Status': 'Not Started',
                            'Priority': task.get('priority', 'Medium'),
                            **task.get('properties', {})
                        }
                    }
                    
                    if inspect.iscoroutinefunction(add_tool):
                        task_result = await add_tool(task_params)
                    else:
                        task_result = add_tool(task_params)
                    
                    if task_result.get('success'):
                        tasks_created += 1
        
        return {
            'success': True,
            'plan_page_id': plan_page_id,
            'tasks_created': tasks_created,
            'milestones': milestones
        }
        
    except Exception as e:
        logger.error(f"Implementation plan creation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _create_plan_from_spec(
    spec_content: str,
    spec_title: str,
    plan_type: str,
    breakdown_level: str,
    include_milestones: bool
) -> tuple:
    """Create implementation plan from specification."""
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if claude_skill:
            generate_tool = claude_skill.tools.get('generate_text_tool')
            
            if generate_tool:
                prompt = f"""Analyze this specification and create an implementation plan:

**Specification:**
{spec_content[:4000]}

**Plan Type:** {plan_type}
**Breakdown Level:** {breakdown_level}
**Include Milestones:** {include_milestones}

Create:
1. Implementation plan with phases/stages
2. List of tasks (breakdown level: {breakdown_level})
3. Milestones with dates (if requested)

Return JSON format:
{{
  "plan": "Implementation plan content",
  "tasks": [
    {{"title": "Task name", "priority": "High/Medium/Low", "properties": {{}}}}
  ],
  "milestones": [
    {{"name": "Milestone name", "date": "YYYY-MM-DD", "description": "..."}}
  ]
}}"""
                
                if inspect.iscoroutinefunction(generate_tool):
                    result = await generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 180
                    })
                else:
                    result = generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 180
                    })
                
                if result.get('success'):
                    import json
                    import re
                    text = result.get('text', '')
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            plan_content = parsed.get('plan', '')
                            tasks = parsed.get('tasks', [])
                            milestones = parsed.get('milestones', [])
                            
                            # Format plan content
                            tasks_section = _format_tasks(tasks)
                            milestones_section = f'## Milestones\n\n{_format_milestones(milestones)}' if milestones else ''
                            
                            formatted_plan = f"""# Implementation Plan: {spec_title}

**Created:** {datetime.now().strftime('%Y-%m-%d')}
**Plan Type:** {plan_type}
**Breakdown Level:** {breakdown_level}

{plan_content}

## Tasks

{tasks_section}

{milestones_section}

---
*Generated from specification*
"""
                            
                            return formatted_plan, tasks, milestones
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        logger.debug(f"AI plan generation failed: {e}")
    
    # Fallback: Simple plan
    newline = "\n"
    plan_content = f"""# Implementation Plan: {spec_title}

## Overview

Implementation plan based on specification.

## Phases

1. **Planning & Setup**
   - Review specification
   - Set up development environment
   - Create initial tasks

2. **Development**
   - Implement core features
   - Add supporting features
   - Integration and testing

3. **Testing & Refinement**
   - Unit testing
   - Integration testing
   - Bug fixes and refinements

4. **Deployment**
   - Final testing
   - Deployment preparation
   - Launch

## Tasks

- [ ] Review specification
- [ ] Set up development environment
- [ ] Implement core features
- [ ] Add supporting features
- [ ] Testing
- [ ] Deployment

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    
    tasks = [
        {'title': 'Review specification', 'priority': 'High'},
        {'title': 'Set up development environment', 'priority': 'High'},
        {'title': 'Implement core features', 'priority': 'High'},
        {'title': 'Add supporting features', 'priority': 'Medium'},
        {'title': 'Testing', 'priority': 'High'},
        {'title': 'Deployment', 'priority': 'High'}
    ]
    
    milestones = [
        {'name': 'Planning Complete', 'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'), 'description': 'Specification reviewed and plan finalized'},
        {'name': 'Development Complete', 'date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'), 'description': 'All features implemented'},
        {'name': 'Testing Complete', 'date': (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d'), 'description': 'All tests passing'},
        {'name': 'Deployment', 'date': (datetime.now() + timedelta(days=28)).strftime('%Y-%m-%d'), 'description': 'Ready for production'}
    ] if include_milestones else []
    
    return plan_content, tasks, milestones


def _format_tasks(tasks: List[Dict]) -> str:
    """Format tasks list."""
    
    if not tasks:
        return "No tasks defined."
    
    formatted = ""
    for i, task in enumerate(tasks, 1):
        title = task.get('title', 'Untitled')
        priority = task.get('priority', 'Medium')
        formatted += f"{i}. **{title}** (Priority: {priority})\n"
    
    return formatted


def _format_milestones(milestones: List[Dict]) -> str:
    """Format milestones list."""
    
    if not milestones:
        return ""
    
    formatted = ""
    for milestone in milestones:
        name = milestone.get('name', 'Untitled')
        date = milestone.get('date', 'TBD')
        description = milestone.get('description', '')
        formatted += f"### {name}\n"
        formatted += f"**Date:** {date}\n"
        if description:
            formatted += f"**Description:** {description}\n"
        formatted += "\n"
    
    return formatted
