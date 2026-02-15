"""
Skill Writer — Creates new Jotty skills from natural language descriptions.

Wraps the core SkillGenerator to provide a skill-level interface for
on-demand skill creation, improvement, and listing.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)
status = SkillStatus("skill-writer")


def _get_generator():
    """Get or create the SkillGenerator singleton."""
    from Jotty.core.capabilities.registry.skill_generator import get_skill_generator
    try:
        from Jotty.core.capabilities.skills import get_registry
        registry = get_registry()
        # SkillsRegistry is the underlying registry object
        skills_registry = getattr(registry, '_skills_registry', None)
    except Exception:
        skills_registry = None
    return get_skill_generator(skills_registry=skills_registry)


def _sanitize_name(name: str) -> str:
    """Normalize a skill name to kebab-case."""
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9\-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name[:60] if name else 'unnamed-skill'


@tool_wrapper(required_params=['name', 'description'])
def create_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new Jotty skill from a natural language description.

    Args:
        params: Dictionary containing:
            - name (str, required): Skill name in kebab-case (e.g., "pdf-merger")
            - description (str, required): What the skill should do
            - requirements (list[str], optional): Python packages needed
            - examples (list[str], optional): Example usage scenarios

    Returns:
        dict with skill_name, skill_path, tools on success, error on failure
    """
    status.set_callback(params.pop('_status_callback', None))

    name = _sanitize_name(params['name'])
    description = params['description']
    requirements = params.get('requirements')
    examples = params.get('examples')

    if not description or len(description) < 10:
        return tool_error(
            'Description must be at least 10 characters. '
            'Example: {"name": "pdf-merger", "description": "Merge multiple PDF files into one"}'
        )

    status.emit("generating", f"Generating skill '{name}'...")

    try:
        generator = _get_generator()
    except Exception as e:
        return tool_error(f'Could not initialize skill generator: {e}')

    # Build requirements string
    req_str = None
    if requirements:
        if isinstance(requirements, list):
            req_str = ', '.join(requirements)
        else:
            req_str = str(requirements)

    try:
        result = generator.generate_skill(
            skill_name=name,
            description=description,
            requirements=req_str,
            examples=examples,
        )
    except Exception as e:
        return tool_error(f'Skill generation failed: {e}')

    status.emit("validating", "Validating generated skill...")

    # Validate
    validation = generator.validate_generated_skill(name)

    # Discover tool function names from generated tools.py
    tools_found: List[str] = []
    tools_py_path = Path(result.get('tools_py', ''))
    if tools_py_path.exists():
        content = tools_py_path.read_text()
        tools_found = re.findall(r'^def (\w+_tool)\s*\(', content, re.MULTILINE)
        if not tools_found:
            tools_found = re.findall(r'^def (\w+)\s*\(', content, re.MULTILINE)

    status.emit("done", f"Skill '{name}' created successfully")

    return tool_response(
        skill_name=name,
        skill_path=result.get('path', ''),
        tools=tools_found,
        reloaded=result.get('reloaded', False),
        valid=validation.get('valid', False),
        validation_errors=validation.get('errors', []),
    )


@tool_wrapper(required_params=['name', 'feedback'])
def improve_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Improve an existing Jotty skill based on feedback.

    Args:
        params: Dictionary containing:
            - name (str, required): Name of the existing skill
            - feedback (str, required): What to change or improve

    Returns:
        dict with skill_name, changes on success, error on failure
    """
    status.set_callback(params.pop('_status_callback', None))

    name = _sanitize_name(params['name'])
    feedback = params['feedback']

    if not feedback or len(feedback) < 5:
        return tool_error(
            'Feedback must be at least 5 characters. '
            'Example: {"name": "my-skill", "feedback": "Add error handling for network failures"}'
        )

    status.emit("improving", f"Improving skill '{name}'...")

    try:
        generator = _get_generator()
    except Exception as e:
        return tool_error(f'Could not initialize skill generator: {e}')

    try:
        result = generator.improve_skill(
            skill_name=name,
            feedback=feedback,
        )
    except ValueError as e:
        return tool_error(str(e))
    except Exception as e:
        return tool_error(f'Skill improvement failed: {e}')

    status.emit("done", f"Skill '{name}' improved")

    return tool_response(
        skill_name=name,
        changes=f"Applied feedback: {feedback}",
        improved=result.get('improved', True),
    )


@tool_wrapper(required_params=[])
def list_skills_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all available skills in the Jotty registry.

    Args:
        params: Dictionary containing:
            - category (str, optional): Filter by category
            - search (str, optional): Search term to filter skills

    Returns:
        dict with skills list and total count
    """
    status.set_callback(params.pop('_status_callback', None))

    category = params.get('category', '').strip().lower()
    search = params.get('search', '').strip().lower()

    status.emit("listing", "Loading skill registry...")

    try:
        from Jotty.core.capabilities.skills import get_registry
        registry = get_registry()
        all_skills = registry.list_skills()
    except Exception as e:
        return tool_error(f'Could not load skill registry: {e}')

    skills_info: List[Dict[str, str]] = []
    for skill in all_skills:
        if isinstance(skill, dict):
            s_name = skill.get('name', '')
            s_desc = skill.get('description', '')
            s_cat = skill.get('category', '')
        elif hasattr(skill, 'name'):
            s_name = getattr(skill, 'name', '')
            s_desc = getattr(skill, 'description', '')
            s_cat = getattr(skill, 'category', '')
        else:
            s_name = str(skill)
            s_desc = ''
            s_cat = ''

        if category and category not in s_cat.lower():
            continue
        if search and search not in s_name.lower() and search not in s_desc.lower():
            continue

        skills_info.append({
            'name': s_name,
            'description': s_desc[:120],
            'category': s_cat,
        })

    return tool_response(
        skills=skills_info,
        total=len(skills_info),
    )


__all__ = ['create_skill_tool', 'improve_skill_tool', 'list_skills_tool']
