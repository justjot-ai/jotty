# Anthropic Best Practices - Quick Reference

**For Jotty Skill Developers**

---

## ✅ The Golden Rules

### 1. Tool Naming
```python
# ✅ GOOD - Clear, semantic, action-oriented
calculate_tool(params)
convert_units_tool(params)
send_telegram_message_tool(params)
fetch_weather_data_tool(params)

# ❌ BAD - Vague, cryptic, generic
calc(params)
conv(params)
send(params)
get_data(params)
```

### 2. Imports (MANDATORY)
```python
# Every tools.py MUST have these imports
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
```

### 3. Tool Decorator (MANDATORY)
```python
# ✅ GOOD - Always use @tool_wrapper
@tool_wrapper(required_params=['query', 'limit'])
def search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for items."""

# ❌ BAD - No decorator, manual validation
def search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    if 'query' not in params:
        return {'success': False, 'error': 'Missing query'}
```

### 4. Error Handling with Examples
```python
# ✅ GOOD - Actionable error with corrective example
return tool_error(
    'Invalid date format. Use ISO 8601: "2024-01-15T10:30:00Z"'
)

return tool_error(
    f'Parameter "count" must be positive integer, got: {count}. '
    f'Example: {{"count": 10}}'
)

# ❌ BAD - Vague, no guidance
return tool_error('Invalid input')
return tool_error('Error')
```

### 5. Success Responses (Semantic Fields Only)
```python
# ✅ GOOD - Clear, semantic field names
return tool_response(
    result=42,
    from_unit='celsius',
    to_unit='fahrenheit',
    conversion_rate=1.8
)

# ❌ BAD - Cryptic, UUIDs, technical IDs
return tool_response(
    r=42,
    id='550e8400-e29b-41d4-a716-446655440000',
    src_u='c',
    dst_u='f'
)
```

### 6. Status Reporting
```python
# Initialize status emitter (module-level)
status = SkillStatus("my-skill")

@tool_wrapper(required_params=['query'])
def search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    # Set callback from params
    status.set_callback(params.pop('_status_callback', None))

    # Emit progress updates
    status.emit("Searching", "🔍 Searching database...")
    results = perform_search()

    status.emit("Processing", "⚙️ Processing results...")
    processed = process_results(results)

    return tool_response(results=processed)
```

### 7. Exports (MANDATORY)
```python
# At end of tools.py
__all__ = ['search_tool', 'filter_tool', 'aggregate_tool']
```

---

## 📋 Complete Tool Template

```python
"""
[Skill Name] Skill Tools

[Brief description of what this skill does]
"""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("skill-name")


@tool_wrapper(required_params=['param1', 'param2'])
def my_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    [Clear description of what this tool does]

    Supports:
    - [Feature 1]
    - [Feature 2]

    Args:
        params: Dictionary containing:
            - param1 (str, required): Description with example
            - param2 (int, optional): Description, defaults to 10

    Returns:
        Dictionary with success, result, error
    """
    # Set status callback
    status.set_callback(params.pop('_status_callback', None))

    # Extract parameters
    param1 = params.get('param1')
    param2 = params.get('param2', 10)

    # Validate
    if not isinstance(param2, int) or param2 <= 0:
        return tool_error(
            f'Parameter "param2" must be positive integer, got: {param2}. '
            f'Example: {{"param2": 10}}'
        )

    # Process with status updates
    status.emit("Processing", "🔄 Processing data...")

    try:
        result = perform_operation(param1, param2)

        return tool_response(
            result=result,
            param1=param1,
            param2=param2
        )

    except ValueError as e:
        return tool_error(
            f'Invalid value: {str(e)}. Use format: "YYYY-MM-DD"'
        )
    except Exception as e:
        return tool_error(f'Operation failed: {str(e)}')


__all__ = ['my_skill_tool']
```

---

## 📝 SKILL.md Template

```yaml
---
name: skillname
description: "Clear description with use cases. Use when the user wants to [action]."
---

# Skill Name

## Description
[Detailed description explaining what this skill does as if to a new team member]

## Type
base  # or derived/composite

## Capabilities
- data-fetch
- analyze
- communicate

## Triggers
- "search for"
- "find"
- "lookup"

## Category
workflow-automation  # or communication, data-analysis, research, general

## Tools

### skill_name_tool
[Clear description]

**Parameters:**
- `param_name` (str, required): Description with example value
- `limit` (int, optional): Maximum results, defaults to 10

**Returns:**
- `success` (bool): Whether operation succeeded
- `result` (list): List of results
- `error` (str, optional): Error message if failed

## Usage Examples
```python
# Example 1: Basic usage
result = skill_name_tool({
    'param_name': 'example'
})

# Example 2: With optional parameters
result = skill_name_tool({
    'param_name': 'example',
    'limit': 20
})
```

## Requirements
[External dependencies, API keys, etc.]

## Error Handling
Common errors and solutions:
- **Invalid format**: Use ISO 8601: "2024-01-15"
- **Missing API key**: Set `API_KEY` environment variable
```

---

## 🚫 Common Pitfalls

### 1. Vague Errors
```python
# ❌ BAD
return tool_error('Invalid input')

# ✅ GOOD
return tool_error(
    'Invalid date format. Expected "YYYY-MM-DD", got: "01/15/2024"'
)
```

### 2. Cryptic Parameter Names
```python
# ❌ BAD
@tool_wrapper(required_params=['q', 'n', 'o'])
def search_tool(params):

# ✅ GOOD
@tool_wrapper(required_params=['query', 'limit', 'offset'])
def search_tool(params):
```

### 3. UUIDs in Responses
```python
# ❌ BAD
return tool_response(
    id='550e8400-e29b-41d4-a716-446655440000',
    result=data
)

# ✅ GOOD
return tool_response(
    result=data,
    user_name='John Doe',  # Semantic, human-readable
    created_at='2024-01-15T10:30:00Z'
)
```

### 4. Missing Decorator
```python
# ❌ BAD - No validation
def my_tool(params):
    if 'query' not in params:
        return tool_error('Missing query')

# ✅ GOOD - Automatic validation
@tool_wrapper(required_params=['query'])
def my_tool(params):
    # query is guaranteed to exist
```

### 5. No Status Updates
```python
# ❌ BAD - Silent execution
def long_running_tool(params):
    result = expensive_operation()  # User sees nothing
    return tool_response(result=result)

# ✅ GOOD - Progress updates
def long_running_tool(params):
    status.set_callback(params.pop('_status_callback', None))

    status.emit("Starting", "🚀 Starting operation...")
    result = expensive_operation()

    status.emit("Finalizing", "✅ Finalizing results...")
    return tool_response(result=result)
```

---

## 🎯 Checklist for New Skills

Before submitting a skill, verify:

- [ ] **Imports**: Has `tool_wrapper`, `tool_response`, `tool_error`
- [ ] **Decorator**: All tools use `@tool_wrapper(required_params=[...])`
- [ ] **Naming**: Function ends with `_tool`, action-oriented
- [ ] **Docstring**: Clear description with Args and Returns
- [ ] **Errors**: Include corrective examples
- [ ] **Responses**: Semantic field names, no UUIDs
- [ ] **Status**: Uses `SkillStatus` for progress
- [ ] **Exports**: Has `__all__ = [...]`
- [ ] **SKILL.md**: Has triggers, capabilities, type
- [ ] **Examples**: SKILL.md includes usage examples

---

## 🔬 Validation

### Automatic Validation
```python
from Jotty.core.registry.skill_generator_improved import get_improved_skill_generator

generator = get_improved_skill_generator()
validation = generator.validate_generated_skill_deep('my-skill')

if not validation['valid']:
    print("❌ Errors:", validation['errors'])

if validation['warnings']:
    print("⚠️ Warnings:", validation['warnings'])
```

### Manual Validation
```bash
# 1. Check imports
grep -q "from Jotty.core.utils.tool_helpers import" skills/my-skill/tools.py

# 2. Check decorator
grep -q "@tool_wrapper" skills/my-skill/tools.py

# 3. Check exports
grep -q "__all__ = " skills/my-skill/tools.py

# 4. Syntax check
python3 -m py_compile skills/my-skill/tools.py
```

---

## 📚 Examples

### Example 1: Simple Skill
```python
"""Calculator Skill"""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

@tool_wrapper(required_params=['expression'])
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform mathematical calculations."""
    try:
        result = eval(params['expression'])
        return tool_response(result=float(result))
    except ZeroDivisionError:
        return tool_error('Division by zero')
    except Exception as e:
        return tool_error(
            f'Invalid expression: {str(e)}. '
            f'Example: "2 + 2" or "sqrt(16)"'
        )

__all__ = ['calculate_tool']
```

### Example 2: Skill with Status
```python
"""Web Search Skill"""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
import requests

status = SkillStatus("web-search")

@tool_wrapper(required_params=['query'])
def web_search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search the web."""
    status.set_callback(params.pop('_status_callback', None))

    query = params['query']
    limit = params.get('limit', 10)

    status.emit("Searching", f"🔍 Searching for: {query}")

    try:
        response = requests.get(f"https://api.example.com/search?q={query}&limit={limit}")
        results = response.json()

        status.emit("Complete", f"✅ Found {len(results)} results")

        return tool_response(
            results=results,
            query=query,
            count=len(results)
        )

    except requests.RequestException as e:
        return tool_error(
            f'Search API failed: {str(e)}. Check network connection.'
        )

__all__ = ['web_search_tool']
```

### Example 3: Composite Skill
```python
"""Research to PDF Skill (Composite)"""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("research-to-pdf")

@async_tool_wrapper(required_params=['topic'])
async def research_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Research topic, create PDF, send to Telegram (all-in-one)."""
    status.set_callback(params.pop('_status_callback', None))

    topic = params['topic']

    # Step 1: Search
    status.emit("Searching", f"🔍 Researching: {topic}")
    from Jotty.core.registry import get_unified_registry
    registry = get_unified_registry()
    web_search = registry.get_skill('web-search').get_tool('web_search_tool')
    search_results = await web_search({'query': topic})

    # Step 2: Summarize
    status.emit("Analyzing", "📊 Analyzing results...")
    llm_tool = registry.get_skill('claude-cli-llm').get_tool('claude_cli_llm_tool')
    summary = await llm_tool({
        'prompt': f"Summarize: {search_results['results']}"
    })

    # Step 3: Create PDF
    status.emit("Creating", "📄 Creating PDF...")
    doc_tool = registry.get_skill('document-converter').get_tool('document_converter_tool')
    pdf = await doc_tool({
        'content': summary['response'],
        'format': 'pdf'
    })

    # Step 4: Send
    status.emit("Sending", "📤 Sending to Telegram...")
    telegram_tool = registry.get_skill('telegram-sender').get_tool('telegram_send_tool')
    await telegram_tool({'file': pdf['path']})

    return tool_response(
        topic=topic,
        pdf_path=pdf['path'],
        summary_length=len(summary['response'])
    )

__all__ = ['research_to_pdf_tool']
```

---

## 🔗 References

- [Anthropic: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Anthropic: Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Claude API: Tool Use Implementation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [Jotty: Full Verification Report](./ANTHROPIC_BEST_PRACTICES_VERIFICATION.md)
- [Jotty: Skill Builder Improvements](./SKILL_BUILDER_PROMPT_IMPROVEMENTS.md)
