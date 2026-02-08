"""
Skill Creator Skill - Help create new Jotty skills.

Generates templates, validates structure, and provides guidance
for creating effective Jotty skills.
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

# Status emitter for progress updates
status = SkillStatus("skill-creator")


logger = logging.getLogger(__name__)


async def create_skill_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new skill template with proper structure.
    
    Args:
        params:
            - skill_name (str): Name of skill (kebab-case)
            - description (str): Brief description
            - output_directory (str, optional): Output directory
            - include_tools (bool, optional): Include tools.py
            - include_requirements (bool, optional): Include requirements.txt
    
    Returns:
        Dictionary with created files and paths
    """
    status.set_callback(params.pop('_status_callback', None))

    skill_name = params.get('skill_name', '')
    description = params.get('description', '')
    output_directory = params.get('output_directory', 'skills')
    include_tools = params.get('include_tools', True)
    include_requirements = params.get('include_requirements', True)
    
    if not skill_name:
        return {
            'success': False,
            'error': 'skill_name is required'
        }
    
    if not description:
        return {
            'success': False,
            'error': 'description is required'
        }
    
    # Validate skill name (kebab-case)
    if not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', skill_name):
        return {
            'success': False,
            'error': 'skill_name must be in kebab-case (e.g., my-skill-name)'
        }
    
    try:
        # Create skill directory
        output_path = Path(os.path.expanduser(output_directory))
        skill_path = output_path / skill_name
        skill_path.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # Create SKILL.md
        skill_md_content = f"""# {skill_name.replace('-', ' ').title()} Skill

{description}

## Description

[Detailed description of what this skill does and when to use it]

## Tools

### `tool_name_tool`

[Tool description]

**Parameters:**
- `param1` (type, required): Description
- `param2` (type, optional): Description

**Returns:**
- `success` (bool): Whether operation succeeded
- `result` (type): Result data
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await tool_name_tool({{
    'param1': 'value1'
}})
```

## Dependencies

- `dependency1`: For feature X
- `dependency2`: For feature Y
"""
        
        skill_md_path = skill_path / 'SKILL.md'
        skill_md_path.write_text(skill_md_content, encoding='utf-8')
        files_created.append('SKILL.md')
        
        # Create tools.py if requested
        if include_tools:
            tools_content = f'''"""
{skill_name.replace('-', ' ').title()} Skill - {description}

[Detailed description of the skill's functionality]
"""
import asyncio
import logging
from typing import Dict, Any

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)


async def tool_name_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    [Tool description]
    
    Args:
        params:
            - param1 (type): Description
            - param2 (type, optional): Description
    
    Returns:
        Dictionary with results
    """
    status.set_callback(params.pop('_status_callback', None))

    param1 = params.get('param1')
    
    if not param1:
        return {{
            'success': False,
            'error': 'param1 is required'
        }}
    
    try:
        # Implementation here
        result = {{'data': 'example'}}
        
        return {{
            'success': True,
            'result': result
        }}
        
    except Exception as e:
        logger.error(f"Tool execution failed: {{e}}", exc_info=True)
        return {{
            'success': False,
            'error': str(e)
        }}
'''
            
            tools_path = skill_path / 'tools.py'
            tools_path.write_text(tools_content, encoding='utf-8')
            files_created.append('tools.py')
        
        # Create requirements.txt if requested
        if include_requirements:
            requirements_content = """# Add your Python dependencies here
# Example:
# requests>=2.31.0
# pandas>=2.0.0
"""
            
            requirements_path = skill_path / 'requirements.txt'
            requirements_path.write_text(requirements_content, encoding='utf-8')
            files_created.append('requirements.txt')
        
        return {
            'success': True,
            'skill_path': str(skill_path),
            'files_created': files_created
        }
        
    except Exception as e:
        logger.error(f"Skill creation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def validate_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an existing skill's structure and metadata.
    
    Args:
        params:
            - skill_path (str): Path to skill directory
    
    Returns:
        Dictionary with validation results
    """
    status.set_callback(params.pop('_status_callback', None))

    skill_path = params.get('skill_path', '')
    
    if not skill_path:
        return {
            'success': False,
            'error': 'skill_path is required'
        }
    
    skill_dir = Path(os.path.expanduser(skill_path))
    
    if not skill_dir.exists():
        return {
            'success': False,
            'error': f'Skill directory not found: {skill_path}'
        }
    
    issues = []
    warnings = []
    
    # Check for SKILL.md
    skill_md = skill_dir / 'SKILL.md'
    if not skill_md.exists():
        issues.append('SKILL.md is missing (required)')
    else:
        # Validate SKILL.md content
        content = skill_md.read_text(encoding='utf-8')
        
        # Check for YAML frontmatter
        if not content.startswith('---'):
            issues.append('SKILL.md missing YAML frontmatter')
        else:
            # Extract frontmatter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                
                # Check for name
                if 'name:' not in frontmatter:
                    issues.append('YAML frontmatter missing "name" field')
                
                # Check for description
                if 'description:' not in frontmatter:
                    issues.append('YAML frontmatter missing "description" field')
    
    # Check for tools.py (optional but recommended)
    tools_py = skill_dir / 'tools.py'
    if not tools_py.exists():
        warnings.append('tools.py not found (recommended for Jotty skills)')
    
    # Check for requirements.txt (optional)
    requirements_txt = skill_dir / 'requirements.txt'
    if not requirements_txt.exists():
        warnings.append('requirements.txt not found (optional but recommended)')
    
    # Validate skill name matches directory name
    if skill_dir.name != skill_dir.name.lower().replace('_', '-'):
        warnings.append(f'Skill directory name should be kebab-case: {skill_dir.name}')
    
    valid = len(issues) == 0
    
    return {
        'success': True,
        'valid': valid,
        'issues': issues,
        'warnings': warnings
    }
