# Skill Builder Prompt Improvements

## Implementation Plan for High-Priority Fixes

### 1. Enhanced SKILL.md Generation Prompt

**File:** `core/registry/skill_generator.py`
**Method:** `_generate_skill_md()`

#### Current Prompt (Line 189-208)
```python
prompt = f"""Create SKILL.md for "{skill_name}" skill.

Description: {description}
{f'Requirements: {requirements}' if requirements else ''}

Output ONLY the markdown file content, no explanations:

# {skill_name}

## Description
{description}

## Tools
[List tools this skill provides]

## Usage
[Usage examples]

## Requirements
{requirements or 'No external dependencies'}"""
```

#### Improved Prompt
```python
prompt = f"""Create SKILL.md for "{skill_name}" skill following Anthropic best practices.

REQUIREMENTS:
1. Clear, unambiguous description explaining what the skill does (as if to a new team member)
2. List all tools with semantic, action-oriented names ending in _tool
3. Document parameters with types and examples
4. Include natural language triggers for tool discovery
5. Specify skill type (base/derived/composite) and category

TEMPLATE:
---
name: {skill_name.replace('-', '_').replace('_', '')}
description: "[Clear description with use cases. Include trigger phrases: 'when the user wants to...']"
---

# {skill_name.title().replace('-', ' ')} Skill

## Description
{description}

## Type
base  # or derived/composite

## Capabilities
- [List capabilities: data-fetch, communicate, analyze, generate, etc.]

## Triggers
- "[Natural language phrase 1]"
- "[Natural language phrase 2]"
- "[Action verb 3]"

## Category
[workflow-automation | communication | data-analysis | research | general]

## Tools

### {skill_name.replace('-', '_')}_tool
[Clear description of what this tool does]

**Parameters:**
- `param_name` (type, required/optional): Description with example value
- `another_param` (type, required/optional): Description

**Returns:**
- `success` (bool): Whether operation succeeded
- `result` (type): The main result
- `error` (str, optional): Error message if failed

[Additional tools if applicable]

## Usage Examples
```python
# Example 1: [Common use case]
result = {skill_name.replace('-', '_')}_tool({{
    'param_name': 'example_value'
}})

# Example 2: [Edge case]
result = {skill_name.replace('-', '_')}_tool({{
    'param_name': 'different_value',
    'another_param': 123
}})
```

## Requirements
{requirements or 'No external dependencies'}

## Error Handling
Common errors and solutions:
- [Error type]: [Corrective action with example]

USER INPUTS:
Description: {description}
Requirements: {requirements or 'None'}
{f'Examples: {examples}' if examples else ''}

Generate the complete SKILL.md following the template above:"""
```

---

### 2. Enhanced tools.py Generation Prompt

**File:** `core/registry/skill_generator.py`
**Method:** `_generate_tools_py()`

#### Current Prompt (Line 278-288)
```python
prompt = f"""Write Python code for tools.py file. Skill: "{skill_name}".

Description: {description}
{f'Requirements: {requirements}' if requirements else ''}

Create Python functions that:
- Accept params: dict parameter
- Return dict with success/error info
- Function names end with _tool

Output ONLY Python code, no explanations, no markdown:"""
```

#### Improved Prompt
```python
prompt = f"""Write production-ready Python code for tools.py following Anthropic best practices.

CRITICAL REQUIREMENTS:

1. **Imports** (MANDATORY):
```python
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
```

2. **Tool Naming**:
   - Function name: `{skill_name.replace('-', '_')}_tool` (action-oriented, semantic)
   - Use clear, unambiguous parameter names (e.g., `user_id` not `uid`)

3. **Tool Decorator** (MANDATORY):
```python
@tool_wrapper(required_params=['param1', 'param2'])
def {skill_name.replace('-', '_')}_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    [Clear description of what this tool does]

    Args:
        params: Dictionary containing:
            - param1 (type): Description
            - param2 (type): Description

    Returns:
        Dictionary with success, result, and optional error
    """
```

4. **Status Reporting**:
```python
status = SkillStatus("{skill_name}")

def my_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    status.set_callback(params.pop('_status_callback', None))
    status.emit("Processing", "🔄 Processing data...")
```

5. **Error Handling with Corrective Examples**:
```python
# BAD - vague error
return tool_error('Invalid input')

# GOOD - actionable error with example
return tool_error(
    'Invalid date format. Use ISO 8601: "2024-01-15T10:30:00Z"'
)

# GOOD - show what went wrong AND how to fix
return tool_error(
    f'Parameter "count" must be positive integer, got: {{params.get("count")}}. '
    f'Example: {{"count": 10}}'
)
```

6. **Success Responses with Semantic Fields**:
```python
# GOOD - semantic field names, no UUIDs
return tool_response(
    result=calculated_value,      # Main result
    from_unit='meters',           # Semantic names
    to_unit='feet',
    conversion_rate=3.28084
)

# BAD - cryptic names, UUIDs
return tool_response(
    r=calculated_value,
    id='550e8400-e29b-41d4-a716-446655440000',
    t='m->ft'
)
```

7. **Parameter Extraction**:
```python
# Extract with defaults and validation
query = params.get('query', '')
limit = params.get('limit', 10)

if not query:
    return tool_error('Parameter "query" is required. Example: {{"query": "search term"}}')

if not isinstance(limit, int) or limit <= 0:
    return tool_error(f'Parameter "limit" must be positive integer, got: {{limit}}')
```

8. **Exports** (MANDATORY):
```python
__all__ = ['{skill_name.replace('-', '_')}_tool']
```

USER REQUIREMENTS:
Skill: {skill_name}
Description: {description}
{f'Requirements: {requirements}' if requirements else ''}
{f'Examples: {examples}' if examples else ''}

Generate COMPLETE, RUNNABLE Python code following ALL patterns above.
Include imports, decorator, error handling with examples, status reporting, and exports.

Output ONLY Python code, no markdown fences:"""
```

---

### 3. Implementation Patch

**File:** `core/registry/skill_generator_improved.py`

```python
"""
Improved Skill Generator with Anthropic Best Practices

Drop-in replacement for skill_generator.py with enhanced prompts.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json
import dspy
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


class ImprovedSkillGenerator:
    """
    AI-powered skill generator with Anthropic best practices built-in.

    Improvements over original:
    - Enhanced prompts with explicit best practices
    - Better error handling guidance
    - Semantic naming enforcement
    - Corrective examples in errors
    - AST-based validation
    """

    def __init__(self, skills_dir: Optional[str] = None, lm: Optional[BaseLM] = None, skills_registry=None):
        """Initialize improved skill generator."""
        # [Same initialization as original]
        if skills_dir is None:
            skills_dir = os.getenv("JOTTY_SKILLS_DIR")

            if not skills_dir:
                current_file = Path(__file__).resolve()
                repo_root = current_file.parent.parent.parent
                repo_skills = repo_root / "skills"
                if repo_skills.exists() or repo_root.name == "Jotty":
                    repo_skills.mkdir(exist_ok=True)
                    skills_dir = str(repo_skills)
                else:
                    home = os.path.expanduser("~")
                    skills_dir = os.path.join(home, "jotty", "skills")

        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.skills_registry = skills_registry

        if lm:
            self.lm = lm
        else:
            try:
                self.lm = dspy.LM()
            except Exception:
                self.lm = None

            if self.lm is None:
                try:
                    from ..foundation.unified_lm_provider import UnifiedLMProvider
                    self.lm = UnifiedLMProvider.configure_default_lm()
                    logger.info("✅ ImprovedSkillGenerator using unified LLM provider")
                except Exception as e:
                    logger.warning(f"⚠️ Could not configure unified LM provider: {e}")
                    raise ValueError(
                        "No LLM available. Provide 'lm' parameter or ensure DSPy is configured"
                    )

    def generate_skill(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str] = None,
        examples: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate a skill following Anthropic best practices."""
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(exist_ok=True)

        # Generate SKILL.md with enhanced prompt
        skill_md_content = self._generate_skill_md_improved(
            skill_name, description, requirements, examples
        )
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md_content)

        # Generate tools.py with enhanced prompt
        tools_py_content = self._generate_tools_py_improved(
            skill_name, description, requirements, examples
        )
        tools_py_path = skill_dir / "tools.py"
        tools_py_path.write_text(tools_py_content)

        logger.info(f"✅ Generated skill: {skill_name} (Anthropic best practices)")

        # Validate generated code
        validation = self.validate_generated_skill_deep(skill_name)
        if not validation['valid']:
            logger.warning(f"⚠️ Validation errors: {validation['errors']}")

        # Auto-reload
        reloaded = False
        if self.skills_registry:
            try:
                self.skills_registry.load_all_skills()
                reloaded = True
                logger.info(f"✅ Skill '{skill_name}' auto-reloaded")
            except Exception as e:
                logger.warning(f"⚠️ Auto-reload failed: {e}")

        return {
            "name": skill_name,
            "description": description,
            "path": str(skill_dir),
            "skill_md": str(skill_md_path),
            "tools_py": str(tools_py_path),
            "reloaded": reloaded,
            "validation": validation,
        }

    def _generate_skill_md_improved(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str],
        examples: Optional[list]
    ) -> str:
        """Generate SKILL.md with Anthropic best practices."""
        prompt = f"""Create SKILL.md for "{skill_name}" skill following Anthropic best practices.

REQUIREMENTS:
1. Clear, unambiguous description explaining what the skill does (as if to a new team member)
2. List all tools with semantic, action-oriented names ending in _tool
3. Document parameters with types and examples
4. Include natural language triggers for tool discovery
5. Specify skill type (base/derived/composite) and category

TEMPLATE:
---
name: {skill_name.replace('-', '_').replace('_', '')}
description: "[Clear description with use cases. Include trigger phrases: 'when the user wants to...']"
---

# {skill_name.title().replace('-', ' ')} Skill

## Description
{description}

## Type
base  # or derived/composite

## Capabilities
- [List capabilities: data-fetch, communicate, analyze, generate, etc.]

## Triggers
- "[Natural language phrase 1]"
- "[Natural language phrase 2]"
- "[Action verb 3]"

## Category
[workflow-automation | communication | data-analysis | research | general]

## Tools

### {skill_name.replace('-', '_')}_tool
[Clear description of what this tool does]

**Parameters:**
- `param_name` (type, required/optional): Description with example value

**Returns:**
- `success` (bool): Whether operation succeeded
- `result` (type): The main result
- `error` (str, optional): Error message if failed

USER INPUTS:
Description: {description}
Requirements: {requirements or 'None'}
{f'Examples: {examples}' if examples else ''}

Generate the complete SKILL.md following the template:"""

        try:
            response = self.lm(prompt=prompt, timeout=180)
            text = self._extract_text(response)
            return self._clean_markdown(text)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._fallback_skill_md(skill_name, description, requirements)

    def _generate_tools_py_improved(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str],
        examples: Optional[list]
    ) -> str:
        """Generate tools.py with Anthropic best practices."""
        tool_func_name = skill_name.replace('-', '_') + '_tool'

        prompt = f"""Write production-ready Python code for tools.py following Anthropic best practices.

CRITICAL REQUIREMENTS:

1. IMPORTS (MANDATORY):
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

2. TOOL DECORATOR (MANDATORY):
@tool_wrapper(required_params=['param1'])
def {tool_func_name}(params: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    {description}

    Args:
        params: Dictionary containing required parameters

    Returns:
        Dictionary with success, result, error
    \"\"\"

3. STATUS REPORTING:
status = SkillStatus("{skill_name}")
status.set_callback(params.pop('_status_callback', None))
status.emit("Processing", "🔄 Processing...")

4. ERROR HANDLING WITH CORRECTIVE EXAMPLES:
return tool_error(
    'Invalid format. Use ISO 8601: "2024-01-15T10:30:00Z"'
)

5. SUCCESS RESPONSES WITH SEMANTIC FIELDS:
return tool_response(
    result=value,
    semantic_field=data  # No UUIDs!
)

6. EXPORTS (MANDATORY):
__all__ = ['{tool_func_name}']

USER REQUIREMENTS:
Skill: {skill_name}
Description: {description}
{f'Requirements: {requirements}' if requirements else ''}

Generate COMPLETE Python code following ALL patterns above:"""

        try:
            response = self.lm(prompt=prompt, timeout=300)
            text = self._extract_text(response)
            return self._clean_python_code(text)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._fallback_tools_py(skill_name, description)

    def validate_generated_skill_deep(self, skill_name: str) -> Dict[str, Any]:
        """Deep validation with AST parsing."""
        import ast

        skill_dir = self.skills_dir / skill_name
        errors = []
        warnings = []

        # Check files exist
        skill_md = skill_dir / "SKILL.md"
        tools_py = skill_dir / "tools.py"

        if not skill_md.exists():
            errors.append("Missing SKILL.md")

        if not tools_py.exists():
            errors.append("Missing tools.py")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Parse AST
        try:
            code = tools_py.read_text()
            tree = ast.parse(code)

            # Check imports
            required_imports = {'tool_response', 'tool_error', 'tool_wrapper'}
            found_imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'tool_helpers' in node.module:
                        for alias in node.names:
                            found_imports.add(alias.name)

            missing = required_imports - found_imports
            if missing:
                warnings.append(f"Missing imports: {missing}")

            # Check decorators
            tool_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.endswith('_tool'):
                        tool_functions.append(node.name)

                        has_wrapper = any(
                            (isinstance(d, ast.Name) and d.id == 'tool_wrapper') or
                            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'tool_wrapper')
                            for d in node.decorator_list
                        )

                        if not has_wrapper:
                            warnings.append(f"Function {node.name} missing @tool_wrapper")

            if not tool_functions:
                errors.append("No tool functions found (must end with _tool)")

        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # Helper methods (same extraction logic as original)
    def _extract_text(self, response) -> str:
        """Extract text from various LLM response formats."""
        if isinstance(response, list) and len(response) > 0:
            return response[0]
        elif hasattr(response, 'completions') and response.completions:
            return response.completions[0].text.strip()
        elif hasattr(response, 'text'):
            return response.text.strip()
        elif isinstance(response, str):
            return response.strip()
        return str(response).strip()

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown fences."""
        if "```markdown" in text:
            start = text.find("```markdown") + 11
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return text.strip()

    def _clean_python_code(self, text: str) -> str:
        """Remove code fences."""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return text.strip()

    def _fallback_skill_md(self, skill_name: str, description: str, requirements: str) -> str:
        """Fallback SKILL.md template."""
        return f"""---
name: {skill_name.replace('-', '')}
description: "{description}"
---

# {skill_name.title().replace('-', ' ')} Skill

## Description
{description}

## Type
base

## Capabilities
- general

## Triggers
- "{skill_name.replace('-', ' ')}"

## Category
general

## Tools

### {skill_name.replace('-', '_')}_tool
{description}

**Parameters:**
- `input` (str, required): Input data

**Returns:**
- `success` (bool): Operation status
- `result` (any): Result data

## Requirements
{requirements or 'No external dependencies'}
"""

    def _fallback_tools_py(self, skill_name: str, description: str) -> str:
        """Fallback tools.py template."""
        tool_name = skill_name.replace('-', '_') + '_tool'
        return f'''"""
{skill_name.title().replace('-', ' ')} Skill Tools
"""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

@tool_wrapper(required_params=['input'])
def {tool_name}(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    {description}

    Args:
        params: Dictionary containing:
            - input (str): Input data

    Returns:
        Dictionary with success and result
    """
    input_data = params.get('input', '')

    if not input_data:
        return tool_error(
            'Parameter "input" is required. '
            'Example: {{"input": "sample data"}}'
        )

    # TODO: Implement logic
    return tool_response(
        result="Processed: " + str(input_data),
        input=input_data
    )

__all__ = ['{tool_name}']
'''


# Singleton instance
_improved_generator: Optional[ImprovedSkillGenerator] = None


def get_improved_skill_generator(
    skills_dir: Optional[str] = None,
    lm: Optional[BaseLM] = None,
    skills_registry=None
) -> ImprovedSkillGenerator:
    """Get singleton improved skill generator instance."""
    global _improved_generator
    if _improved_generator is None:
        _improved_generator = ImprovedSkillGenerator(skills_dir, lm, skills_registry)
    elif skills_registry and not _improved_generator.skills_registry:
        _improved_generator.skills_registry = skills_registry
    return _improved_generator
```

---

## Testing Plan

### 1. Unit Tests

**File:** `tests/test_improved_skill_generator.py`

```python
import pytest
from Jotty.core.registry.skill_generator_improved import ImprovedSkillGenerator

@pytest.mark.asyncio
async def test_generated_skill_has_tool_wrapper():
    """Verify generated code uses @tool_wrapper decorator."""
    generator = ImprovedSkillGenerator()

    result = generator.generate_skill(
        skill_name="test-skill",
        description="A test skill"
    )

    tools_py = Path(result['tools_py']).read_text()
    assert '@tool_wrapper' in tools_py
    assert 'tool_response' in tools_py
    assert 'tool_error' in tools_py

@pytest.mark.asyncio
async def test_generated_errors_have_examples():
    """Verify error messages include corrective examples."""
    generator = ImprovedSkillGenerator()

    result = generator.generate_skill(
        skill_name="date-parser",
        description="Parse dates"
    )

    tools_py = Path(result['tools_py']).read_text()
    # Should have error example
    assert 'Example:' in tools_py or 'example:' in tools_py

@pytest.mark.asyncio
async def test_ast_validation():
    """Verify AST-based validation catches issues."""
    generator = ImprovedSkillGenerator()

    # Generate skill
    generator.generate_skill("test-skill", "Test")

    # Validate
    validation = generator.validate_generated_skill_deep("test-skill")

    assert validation['valid'] == True or len(validation['warnings']) > 0
```

### 2. Integration Test

```bash
# Generate a real skill and verify it works
python3 -c "
from Jotty.core.registry.skill_generator_improved import get_improved_skill_generator
from Jotty.core.registry import get_unified_registry

registry = get_unified_registry()
generator = get_improved_skill_generator(skills_registry=registry)

result = generator.generate_skill(
    skill_name='weather-api',
    description='Fetch weather data from OpenWeather API',
    requirements='API key required'
)

print('Generated:', result)
print('Validation:', result['validation'])
"
```

---

## Rollout Plan

### Phase 1: Testing (1 day)
1. Implement `skill_generator_improved.py`
2. Run unit tests
3. Generate 3-5 test skills
4. Compare with manually-written skills

### Phase 2: Deployment (1 day)
1. Backup existing `skill_generator.py`
2. Replace with improved version
3. Re-generate 5-10 existing skills
4. Compare quality

### Phase 3: Documentation (1 day)
1. Update CLAUDE.md with new patterns
2. Document AST validation
3. Create skill development guide

---

## Success Metrics

### Before (Current)
- ❌ Generated skills may lack imports
- ❌ Error messages are vague
- ❌ No decorator enforcement
- ⚠️ Basic validation (file exists)

### After (Improved)
- ✅ All generated skills have proper imports
- ✅ Error messages include corrective examples
- ✅ @tool_wrapper always used
- ✅ AST validation catches structural issues
- ✅ Anthropic best practices embedded in prompts

---

## Migration Guide

### For Existing Skills

```bash
# Regenerate an existing skill with improvements
python3 -c "
from Jotty.core.registry.skill_generator_improved import get_improved_skill_generator

generator = get_improved_skill_generator()

# Read existing SKILL.md to get description
existing_md = Path('skills/calculator/SKILL.md').read_text()

# Re-generate with improved prompts
generator.generate_skill(
    skill_name='calculator',
    description='Mathematical calculations and unit conversions',
    requirements='No external dependencies'
)
"
```

### For New Skills

```python
from Jotty.core.registry.skill_generator_improved import get_improved_skill_generator

generator = get_improved_skill_generator()

result = generator.generate_skill(
    skill_name="my-new-skill",
    description="What it does",
    requirements="API keys, etc.",
    examples=["example 1", "example 2"]
)

print("✅ Generated with Anthropic best practices!")
print("Validation:", result['validation'])
```
